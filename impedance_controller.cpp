#include "open_manipulator_controllers/impedance_controller.h"

namespace open_manipulator_controllers {

bool ImpedanceController::init(hardware_interface::RobotHW* hw,
                               ros::NodeHandle& nh) {
  // List of controlled joints
  if (!nh.getParam("joint_names", joint_names_)) {
    ROS_ERROR("[ImpedanceController] Could not parse joint names");
    return false;
  }
  n_joints_ = joint_names_.size();
  if (n_joints_ == 0) {
    ROS_ERROR("[ImpedanceController] List of joint names is empty.");
    return false;
  }

  // Joint interface
  effort_joint_interface_ = hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface_ == nullptr) {
    ROS_ERROR(
        "[ImpedanceController] Could not get effort joint interface "
        "from hardware!");
    return false;
  }

  // Joint handle
  for (size_t i = 0; i < n_joints_; i++) {
    try {
      effort_joint_handles_.push_back(
          effort_joint_interface_->getHandle(joint_names_[i]));
    } catch (const hardware_interface::HardwareInterfaceException& e) {
      ROS_ERROR_STREAM(
          "[ImpedanceController] Could not get joint handle: " << e.what());
      return false;
    }
  }

  // URDF
  urdf::Model urdf;
  if (!urdf.initParam("robot_description")) {
    ROS_ERROR("[ImpedanceController] Failed to parse urdf file");
    return false;
  }

  // KDL Parser
  if (!kdl_parser::treeFromUrdfModel(urdf, kdl_tree_)) {
    ROS_ERROR("[ImpedanceController] Failed to construct kdl tree");
    return false;
  }

  // KDL Chain
  if (!nh.getParam("root_link", root_name_)) {
    ROS_ERROR("[ImpedanceController] Could not find root link name");
    return false;
  }
  if (!nh.getParam("tip_link", tip_name_)) {
    ROS_ERROR("[ImpedanceController] Could not find tip link name");
    return false;
  }
  if (!kdl_tree_.getChain(root_name_, tip_name_, kdl_chain_)) {
    ROS_ERROR_STREAM(
        "[ImpedanceController] Failed to get KDL chain from tree: ");
    ROS_ERROR_STREAM("  " << root_name_ << " --> " << tip_name_);
    ROS_ERROR_STREAM("  Tree has " << kdl_tree_.getNrOfJoints() << " joints");
    ROS_ERROR_STREAM("  Tree has " << kdl_tree_.getNrOfSegments()
                                   << " segments");
    ROS_ERROR_STREAM("  The segments are:");

    KDL::SegmentMap segment_map = kdl_tree_.getSegments();
    KDL::SegmentMap::iterator it;

    for (it = segment_map.begin(); it != segment_map.end(); it++)
      ROS_ERROR_STREAM("    " << (*it).first);

    return false;
  }

  // Obtain other values (TODO)
  int num = 1.5;
  Kd_ = Eigen::Matrix3d::Identity() * num * num;
  Bd_ = Eigen::Matrix3d::Identity() * num * 2;

  // Resize Variables
  q_.resize(n_joints_);
  q_dot_.resize(n_joints_);
  M_.resize(n_joints_);  // Optional
  C_.resize(n_joints_);  // Optional
  G_.resize(n_joints_);  // Optional
  pos_vel_.resize(n_joints_);
  tau_.resize(n_joints_);

  J_.resize(n_joints_);
  J_pos_.resize(3, n_joints_);
  J_dot_.resize(n_joints_);
  J_pos_dot_.resize(3, n_joints_);

  J_pos_pinv_.resize(n_joints_, 3);

  desired_x_ << 0.286, 0.0, 0.2045;
  desired_x_dot_ << 0, 0, 0;

  // Initialize Solvers (use kdl_chain_)
  KDL::Vector g(0.0, 0.0, -9.81);
  grav_ = g;
  MCG_solver_.reset(new KDL::ChainDynParam(kdl_chain_, grav_));
  Fk_vel_solver_.reset(new KDL::ChainFkSolverVel_recursive(kdl_chain_));
  Jac_solver_.reset(new KDL::ChainJntToJacSolver(kdl_chain_));
  Jac_dot_solver_.reset(new KDL::ChainJntToJacDotSolver(kdl_chain_));

  ROS_INFO("[ImpedanceController] Succesfully Initialized Controller");
  return true;
}

void ImpedanceController::starting(const ros::Time& time) {}

void ImpedanceController::update(const ros::Time& time,
                                 const ros::Duration& period) {
  // Get current joint_states
  for (size_t i = 0; i < kdl_chain_.getNrOfJoints(); i++) {
    q_(i) = effort_joint_handles_[i].getPosition();
    q_dot_(i) = effort_joint_handles_[i].getVelocity();
  }
  pos_vel_.q = q_;
  pos_vel_.qdot = q_dot_;

  // Compute stuff
  // M, C y g
  {
    MCG_solver_->JntToMass(pos_vel_.q, M_);
    MCG_solver_->JntToCoriolis(pos_vel_.q, pos_vel_.qdot, C_);
    MCG_solver_->JntToGravity(pos_vel_.q, G_);
  }
  // fkine X and X dot
  {
    Fk_vel_solver_->JntToCart(pos_vel_, frame_vel_);
    x_ = frame_vel_.p.p;
    x_dot_ = frame_vel_.p.v;
    x_eigen_ << x_.x(), x_.y(), x_.z();
    x_dot_eigen_ << x_dot_.x(), x_dot_.y(), x_dot_.z();
  }
  // Position Jac and Jac dot
  {
    Jac_solver_->JntToJac(pos_vel_.q, J_);
    J_pos_ = J_.data(Eigen::seq(0, 2), Eigen::all);
    Jac_dot_solver_->JntToJacDot(pos_vel_, J_dot_);
    J_pos_dot_ = J_dot_.data(Eigen::seq(0, 2), Eigen::all);
  }

  // Inverse Jacobian
  { J_pos_pinv_ = J_pos_.completeOrthogonalDecomposition().pseudoInverse(); }

  // Desired Inertial matrix
  Md_ = J_pos_pinv_.transpose() * M_.data * J_pos_pinv_;

  auto imposicion = Md_.inverse() * (Bd_ * (desired_x_dot_ - x_dot_eigen_) +
                                     Kd_ * (desired_x_ - x_eigen_));

  auto dinamica = desired_x_dot_dot_ - J_pos_dot_ * q_dot_.data + imposicion;

  tau_.data = M_.data * J_pos_pinv_ * dinamica + C_.data + G_.data;

  // ROS_INFO_STREAM('\n' << J_pos_dot_ * q_dot_.data);

  // for (size_t i = 0; i < kdl_chain_.getNrOfJoints(); i++) {
  //   tau_(i) = 0;
  //   tau_(i) += G_(i);
  // }

  // Set effort
  for (size_t i = 0; i < kdl_chain_.getNrOfJoints(); i++) {
    effort_joint_handles_[i].setCommand(tau_(i));
  }
}

void ImpedanceController::stopping(const ros::Time& time) {}

}  // namespace open_manipulator_controllers

PLUGINLIB_EXPORT_CLASS(open_manipulator_controllers::ImpedanceController,
                       controller_interface::ControllerBase)