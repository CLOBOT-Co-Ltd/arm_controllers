#include <array>
#include <chrono>
#include <iostream>
#include <thread>
#include <cmath>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <arm_interfaces/action/gesture.hpp> // Custom action message

static const std::string kTopicArmSDK = "rt/arm_sdk";
static const std::string kTopicState = "rt/lowstate";
constexpr float Pi = 3.141592654;
constexpr float Pi_2 = 1.57079632;
constexpr double deg_30 = 0.523599;
constexpr double deg_20 = 0.349066;
constexpr double deg_10 = 0.174533;

constexpr int INIT_POS_MOTION = 0;
constexpr int WAVE_HAND_MOTION = 1;
constexpr int FOLLOW_ME_MOTION = 2;
constexpr int ITS_ME_MOTION = 3;
constexpr int INTRODUCE_MOTION = 4;


enum JointIndex
{
  // Left leg
  kLeftHipYaw = 0,
  kLeftHipPitch = 1,
  kLeftHipRoll = 2,
  kLeftKnee = 3,
  kLeftAnkle = 4,
  kLeftAnkleRoll = 5,
  // Right leg
  kRightHipYaw = 6,
  kRightHipPitch = 7,
  kRightHipRoll = 8,
  kRightKnee = 9,
  kRightAnkle = 10,
  kRightAnkleRoll = 11,

  kWaistYaw = 12,

  // Left arm
  kLeftShoulderPitch = 13,
  kLeftShoulderRoll = 14,
  kLeftShoulderYaw = 15,
  kLeftElbow = 16, // left elbow pitch
  kLeftWristRoll = 17, // left elbow roll
  kLeftWristPitch = 18,
  kLeftWristYaw = 19,
  // Right arm
  kRightShoulderPitch = 20,
  kRightShoulderRoll = 21,
  kRightShoulderYaw = 22,
  kRightElbow = 23, // right elbow pitch
  kRightWristRoll = 24, // right elbow roll
  kRightWristPitch = 25,
  kRightWristYaw = 26,

  kNotUsedJoint = 27,
  kNotUsedJoint1 = 28,
  kNotUsedJoint2 = 29,
  kNotUsedJoint3 = 30,
  kNotUsedJoint4 = 31,
  kNotUsedJoint5 = 32,
  kNotUsedJoint6 = 33,
  kNotUsedJoint7 = 34
};

class ArmComtrollerNode : public rclcpp::Node
{
private:
  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> arm_sdk_publisher_;
  unitree_hg::msg::dds_::LowCmd_ msg_;

  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> low_state_subscriber_;
  unitree_hg::msg::dds_::LowState_ state_msg_;   // Member variable to store latest state

  rclcpp_action::Server<arm_interfaces::action::Gesture>::SharedPtr action_server_;

  std::atomic<bool> is_action_active_{false};

  std::array<float, 15> init_pos_;
  std::array<float, 15> wave_hand_init_pos_;
  std::array<float, 15> follow_init_pos_;
  std::array<float, 15> its_me_init_pos_;
  std::array<float, 15> introduce_init_pos_;

  std::array<JointIndex, 15> arm_joints_ = {
    JointIndex::kLeftShoulderPitch, JointIndex::kLeftShoulderRoll,
    JointIndex::kLeftShoulderYaw, JointIndex::kLeftElbow,
    JointIndex::kLeftWristRoll, JointIndex::kLeftWristPitch, JointIndex::kLeftWristYaw,
    JointIndex::kRightShoulderPitch, JointIndex::kRightShoulderRoll,
    JointIndex::kRightShoulderYaw, JointIndex::kRightElbow,
    JointIndex::kRightWristRoll, JointIndex::kRightWristPitch, JointIndex::kRightWristYaw,
    JointIndex::kWaistYaw};

  // Control parameters (can be made parameters if needed)
  std::array<float, 15> kp_array_ = {120, 120, 80, 50, 50, 50, 50,
    120, 120, 80, 50, 50, 50, 50,
    200};
  std::array<float, 15> kd_array_ = {2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0,
    2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0,
    2.0};

  float dq_ = 0.f;
  float tau_ff_ = 0.f;
  float control_dt_ = 0.02f;   // 50 Hz
  float max_joint_velocity_ = 0.75f;   // rad/s
  float max_joint_delta_ = max_joint_velocity_ * control_dt_;   // max angle change per control step

public:
  ArmComtrollerNode(const char * networkInterface)
  : Node("arm_controller_node")
  {
    RCLCPP_INFO(this->get_logger(), "ArmControllerNode started");

    // Initialize Unitree SDK/DDS
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);

    arm_sdk_publisher_.reset(
      new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(
        kTopicArmSDK));
    arm_sdk_publisher_->InitChannel();

    low_state_subscriber_.reset(
      new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
        kTopicState));
    low_state_subscriber_->InitChannel(
      std::bind(
        &ArmComtrollerNode::on_lowstate_received, this,
        std::placeholders::_1), 1);

    // Define target positions
    // Initial position (from h1_2_arm_sdk_dds_example.cpp)
    init_pos_ = {deg_10, 0.3, 0.f, 0, 0, 0, 0,
      deg_10, -0.3, 0.f, 0, 0, 0, 0,
      0.f};

    // Target position (from h1_2_arm_sdk_dds_example.cpp - arms up)
    wave_hand_init_pos_ = {Pi_2 / 2, deg_20, 0, Pi_2 / 2, 0, 0, 0,
      -Pi_2 / 2, -deg_20, 0, -Pi_2 / 2, -Pi_2, 0, 0,
      0};

    follow_init_pos_ = {Pi_2 / 2, deg_20, 0, Pi_2 / 2, 0, 0, 0,
      -deg_30, 0, deg_10, 0, Pi_2, 0, 0,
      0};

    its_me_init_pos_ = {Pi_2 / 2, deg_20, 0, Pi_2 / 2, 0, 0, 0,
      -Pi_2 / 2, -deg_20, 0, -Pi_2 / 2, -Pi_2, 0, 0,
      0};

    introduce_init_pos_ = {-deg_10, deg_30, Pi_2 / 2, deg_30 * 2, 0, 0, 0,
      deg_10, -0.3, 0.f, deg_30 * 2, 0, 0, 0,
      0.f};


    // Initial arm setup (move to init_pos once at startup)
    // initialize_arm();

    // Create the action server
    action_server_ = rclcpp_action::create_server<arm_interfaces::action::Gesture>(
      this->get_node_base_interface(),
      this->get_node_clock_interface(),
      this->get_node_logging_interface(),
      this->get_node_waitables_interface(),
      "gesture",       // The action name
      std::bind(
        &ArmComtrollerNode::handle_goal, this, std::placeholders::_1,
        std::placeholders::_2),
      std::bind(&ArmComtrollerNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&ArmComtrollerNode::handle_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Arm action server started.");
  }

private:
  // Callback for DDS lowstate messages
  void on_lowstate_received(const void * msg)
  {
    auto s = (const unitree_hg::msg::dds_::LowState_ *)msg;
    memcpy(&state_msg_, s, sizeof( unitree_hg::msg::dds_::LowState_ ) );
  }

  // Move arm to a target position over a duration
  void move_arm_to_pose(
    int action_type,
    const std::array<float, 15> & target_pose,
    float duration_sec,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<arm_interfaces::action::Gesture>>
    goal_handle,
    const std::shared_ptr<arm_interfaces::action::Gesture::Feedback> feedback)     // Accept feedback pointer
  {
    RCLCPP_INFO(this->get_logger(), "Moving arm to target pose over %f seconds", duration_sec);
    rclcpp::Rate loop_rate(1.0f / control_dt_);

    std::array<float, 15> start_pos;

    int num_time_steps = static_cast<int>(duration_sec / control_dt_);
    if (num_time_steps == 0) {num_time_steps = 1;}

    // Get current position from state_msg (ensure state_msg is fresh)
    // A more robust approach would wait for a state message if needed
    for (int i = 0; i < arm_joints_.size(); ++i) {
      start_pos.at(i) = state_msg_.motor_state().at(arm_joints_.at(i)).q();
    }


    for (int i = 0; i <= num_time_steps; ++i) {
      if (goal_handle && goal_handle->is_canceling()) {
        RCLCPP_INFO(this->get_logger(), "Action canceled during movement");
        return;         // Exit movement loop
      }

      float phase = static_cast<float>(i) / num_time_steps;
      float smooth_phase = 0.5f - 0.5f * cos(Pi * phase);
      std::array<float, 15> current_jpos_des;

      // sin 이징 interpolation
      for (int j = 0; j < arm_joints_.size(); ++j) {
        current_jpos_des.at(j) = start_pos.at(j) * (1.0f - smooth_phase) + target_pose.at(j) *
          smooth_phase;

        // Set control commands
        msg_.motor_cmd().at(arm_joints_.at(j)).q(current_jpos_des.at(j));
        msg_.motor_cmd().at(arm_joints_.at(j)).dq(dq_);         // Velocity command (usually 0 for position control)
        msg_.motor_cmd().at(arm_joints_.at(j)).kp(kp_array_.at(j));
        msg_.motor_cmd().at(arm_joints_.at(j)).kd(kd_array_.at(j));
        msg_.motor_cmd().at(arm_joints_.at(j)).tau(tau_ff_);         // Feedforward torque (usually 0)

        // Populate feedback (using actual state, not desired)
        if (feedback) {         // Use the passed feedback pointer
          // Ensure feedback->current_joint_angle has enough space or is resized
          // Assuming it's a dynamic container or properly sized
          // For arm_interfaces::action::Gesture feedback structure (JointInfo[]),
          // we need to populate individual JointInfo messages.
          // This part requires more detailed knowledge of JointInfo struct.
          // Based on the image, JointInfo has name, number, angle_rad
          if (feedback->current_joint_angle.size() <= j) {
            feedback->current_joint_angle.resize(j + 1);
          }
          feedback->current_joint_angle.at(j).joint_idx.name = "joint_" +
            std::to_string(arm_joints_.at(j));                                                                        // Example name
          feedback->current_joint_angle.at(j).joint_idx.number = arm_joints_.at(j);
          feedback->current_joint_angle.at(j).angle_rad =
            state_msg_.motor_state().at(arm_joints_.at(j)).q();                                                        // Use actual state for feedback
        }
      }

      // Set weight (if needed for arm control mode, e.g., to enable torque control)
      // Assuming JointIndex::kNotUsedJoint is used for weight based on h1_2_arm_sdk_dds_example.cpp
      msg_.motor_cmd().at(JointIndex::kNotUsedJoint).q(1.0f);   // Example: set weight to 1.0 during movement

      // Send dds msg
      arm_sdk_publisher_->Write(msg_);

      // Publish feedback
      if (goal_handle && feedback) {       // Check both pointers are valid
        goal_handle->publish_feedback(feedback);        // Pass the shared pointer directly
      }

      loop_rate.sleep();
    }


    if (action_type == WAVE_HAND_MOTION) {
      // Get current position from state_msg (ensure state_msg is fresh)
      // A more robust approach would wait for a state message if needed

      std::array<float, 15> current_jpos_des;
      for (int i = 0; i < arm_joints_.size(); ++i) {
        start_pos.at(i) = state_msg_.motor_state().at(arm_joints_.at(i)).q();

        if (i == 9) {
          current_jpos_des.at(i) = start_pos.at(i);
        }
      }

      for (int i = 0; i <= num_time_steps; ++i) {
        if (goal_handle && goal_handle->is_canceling()) {
          RCLCPP_INFO(this->get_logger(), "Action canceled during movement");
          return;       // Exit movement loop
        }

        float phase = static_cast<float>(i) / num_time_steps;
        float smooth_phase = 0.5f - 0.5f * cos(Pi * phase);

        // linear interpolation
        for (int j = 0; j < arm_joints_.size(); ++j) {
          // kRightShoulderYaw
          if (j == 9) {
            current_jpos_des.at(j) +=
              std::clamp(
              (float) (deg_30 * std::sin(4 * Pi * phase)) - current_jpos_des.at(j),
              -max_joint_delta_, max_joint_delta_);
          } else {
            current_jpos_des.at(j) = start_pos.at(j) * (1.0f - smooth_phase) + target_pose.at(j) *
              smooth_phase;
          }

          // Set control commands
          msg_.motor_cmd().at(arm_joints_.at(j)).q(current_jpos_des.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).dq(dq_);       // Velocity command (usually 0 for position control)
          msg_.motor_cmd().at(arm_joints_.at(j)).kp(kp_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).kd(kd_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).tau(tau_ff_);       // Feedforward torque (usually 0)

          // Populate feedback (using actual state, not desired)
          if (feedback) {       // Use the passed feedback pointer
            // Ensure feedback->current_joint_angle has enough space or is resized
            // Assuming it's a dynamic container or properly sized
            // For arm_interfaces::action::Gesture feedback structure (JointInfo[]),
            // we need to populate individual JointInfo messages.
            // This part requires more detailed knowledge of JointInfo struct.
            // Based on the image, JointInfo has name, number, angle_rad
            if (feedback->current_joint_angle.size() <= j) {
              feedback->current_joint_angle.resize(j + 1);
            }
            feedback->current_joint_angle.at(j).joint_idx.name = "joint_" +
              std::to_string(arm_joints_.at(j));                                                                      // Example name
            feedback->current_joint_angle.at(j).joint_idx.number = arm_joints_.at(j);
            feedback->current_joint_angle.at(j).angle_rad =
              state_msg_.motor_state().at(arm_joints_.at(j)).q();                                                      // Use actual state for feedback
          }
        }

        // Set weight (if needed for arm control mode, e.g., to enable torque control)
        // Assuming JointIndex::kNotUsedJoint is used for weight based on h1_2_arm_sdk_dds_example.cpp
        msg_.motor_cmd().at(JointIndex::kNotUsedJoint).q(1.0f);     // Example: set weight to 1.0 during movement

        // Send dds msg
        arm_sdk_publisher_->Write(msg_);

        // Publish feedback
        if (goal_handle && feedback) {     // Check both pointers are valid
          goal_handle->publish_feedback(feedback);      // Pass the shared pointer directly
        }

        loop_rate.sleep();
      }


      // Get current position from state_msg (ensure state_msg is fresh)
      // A more robust approach would wait for a state message if needed
      for (int i = 0; i < arm_joints_.size(); ++i) {
        start_pos.at(i) = state_msg_.motor_state().at(arm_joints_.at(i)).q();
      }


      for (int i = 0; i <= num_time_steps; ++i) {
        if (goal_handle && goal_handle->is_canceling()) {
          RCLCPP_INFO(this->get_logger(), "Action canceled during movement");
          return;       // Exit movement loop
        }

        float phase = static_cast<float>(i) / num_time_steps;
        float smooth_phase = 0.5f - 0.5f * cos(Pi * phase);

        // sin 이징 interpolation
        for (int j = 0; j < arm_joints_.size(); ++j) {


          current_jpos_des.at(j) = start_pos.at(j) * (1.0f - smooth_phase) + init_pos_.at(j) *
            smooth_phase;

          // Set control commands
          msg_.motor_cmd().at(arm_joints_.at(j)).q(current_jpos_des.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).dq(dq_);       // Velocity command (usually 0 for position control)
          msg_.motor_cmd().at(arm_joints_.at(j)).kp(kp_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).kd(kd_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).tau(tau_ff_);       // Feedforward torque (usually 0)

          // Populate feedback (using actual state, not desired)
          if (feedback) {       // Use the passed feedback pointer
            // Ensure feedback->current_joint_angle has enough space or is resized
            // Assuming it's a dynamic container or properly sized
            // For arm_interfaces::action::Gesture feedback structure (JointInfo[]),
            // we need to populate individual JointInfo messages.
            // This part requires more detailed knowledge of JointInfo struct.
            // Based on the image, JointInfo has name, number, angle_rad
            if (feedback->current_joint_angle.size() <= j) {
              feedback->current_joint_angle.resize(j + 1);
            }
            feedback->current_joint_angle.at(j).joint_idx.name = "joint_" +
              std::to_string(arm_joints_.at(j));                                                                      // Example name
            feedback->current_joint_angle.at(j).joint_idx.number = arm_joints_.at(j);
            feedback->current_joint_angle.at(j).angle_rad =
              state_msg_.motor_state().at(arm_joints_.at(j)).q();                                                      // Use actual state for feedback
          }
        }

        // Set weight (if needed for arm control mode, e.g., to enable torque control)
        // Assuming JointIndex::kNotUsedJoint is used for weight based on h1_2_arm_sdk_dds_example.cpp
        msg_.motor_cmd().at(JointIndex::kNotUsedJoint).q(1.0f); // Example: set weight to 1.0 during movement

        // Send dds msg
        arm_sdk_publisher_->Write(msg_);

        // Publish feedback
        if (goal_handle && feedback) {     // Check both pointers are valid
          goal_handle->publish_feedback(feedback);      // Pass the shared pointer directly
        }

        loop_rate.sleep();
      }
    } else if (action_type == FOLLOW_ME_MOTION) {
      // Get current position from state_msg (ensure state_msg is fresh)
      // A more robust approach would wait for a state message if needed

      std::array<float, 15> current_jpos_des;
      for (int i = 0; i < arm_joints_.size(); ++i) {
        start_pos.at(i) = state_msg_.motor_state().at(arm_joints_.at(i)).q();

        if (i == 10) { // kRightElbowPitch
          current_jpos_des.at(i) = start_pos.at(i);
        }
      }

      for (int i = 0; i <= num_time_steps; ++i) {
        if (goal_handle && goal_handle->is_canceling()) {
          RCLCPP_INFO(this->get_logger(), "Action canceled during movement");
          return;       // Exit movement loop
        }

        float phase = static_cast<float>(i) / num_time_steps;
        float smooth_phase = 0.5f - 0.5f * cos(Pi * phase);

        // linear interpolation
        for (int j = 0; j < arm_joints_.size(); ++j) {
          // kRightElbowPitch
          if (j == 10) {
            current_jpos_des.at(j) +=
              std::clamp(
              (float) -(deg_30 / 2 * std::sin(4 * Pi * (phase - Pi_2)) + (deg_30 / 2) ) -
              current_jpos_des.at(j),
              -max_joint_delta_, max_joint_delta_);
          } else {
            current_jpos_des.at(j) = start_pos.at(j) * (1.0f - smooth_phase) + target_pose.at(j) *
              smooth_phase;
          }

          // Set control commands
          msg_.motor_cmd().at(arm_joints_.at(j)).q(current_jpos_des.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).dq(dq_);       // Velocity command (usually 0 for position control)
          msg_.motor_cmd().at(arm_joints_.at(j)).kp(kp_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).kd(kd_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).tau(tau_ff_);       // Feedforward torque (usually 0)

          // Populate feedback (using actual state, not desired)
          if (feedback) {       // Use the passed feedback pointer
            // Ensure feedback->current_joint_angle has enough space or is resized
            // Assuming it's a dynamic container or properly sized
            // For arm_interfaces::action::Gesture feedback structure (JointInfo[]),
            // we need to populate individual JointInfo messages.
            // This part requires more detailed knowledge of JointInfo struct.
            // Based on the image, JointInfo has name, number, angle_rad
            if (feedback->current_joint_angle.size() <= j) {
              feedback->current_joint_angle.resize(j + 1);
            }
            feedback->current_joint_angle.at(j).joint_idx.name = "joint_" +
              std::to_string(arm_joints_.at(j));                                                                      // Example name
            feedback->current_joint_angle.at(j).joint_idx.number = arm_joints_.at(j);
            feedback->current_joint_angle.at(j).angle_rad =
              state_msg_.motor_state().at(arm_joints_.at(j)).q();                                                      // Use actual state for feedback
          }
        }

        // Set weight (if needed for arm control mode, e.g., to enable torque control)
        // Assuming JointIndex::kNotUsedJoint is used for weight based on h1_2_arm_sdk_dds_example.cpp
        msg_.motor_cmd().at(JointIndex::kNotUsedJoint).q(1.0f);     // Example: set weight to 1.0 during movement

        // Send dds msg
        arm_sdk_publisher_->Write(msg_);

        // Publish feedback
        if (goal_handle && feedback) {     // Check both pointers are valid
          goal_handle->publish_feedback(feedback);      // Pass the shared pointer directly
        }

        loop_rate.sleep();
      }


      // Get current position from state_msg (ensure state_msg is fresh)
      // A more robust approach would wait for a state message if needed
      for (int i = 0; i < arm_joints_.size(); ++i) {
        start_pos.at(i) = state_msg_.motor_state().at(arm_joints_.at(i)).q();
      }


      for (int i = 0; i <= num_time_steps; ++i) {
        if (goal_handle && goal_handle->is_canceling()) {
          RCLCPP_INFO(this->get_logger(), "Action canceled during movement");
          return;       // Exit movement loop
        }

        float phase = static_cast<float>(i) / num_time_steps;
        float smooth_phase = 0.5f - 0.5f * cos(Pi * phase);

        // sin 이징 interpolation
        for (int j = 0; j < arm_joints_.size(); ++j) {
          current_jpos_des.at(j) = start_pos.at(j) * (1.0f - smooth_phase) + init_pos_.at(j) *
            smooth_phase;

          // Set control commands
          msg_.motor_cmd().at(arm_joints_.at(j)).q(current_jpos_des.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).dq(dq_);       // Velocity command (usually 0 for position control)
          msg_.motor_cmd().at(arm_joints_.at(j)).kp(kp_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).kd(kd_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).tau(tau_ff_);       // Feedforward torque (usually 0)

          // Populate feedback (using actual state, not desired)
          if (feedback) {       // Use the passed feedback pointer
            // Ensure feedback->current_joint_angle has enough space or is resized
            // Assuming it's a dynamic container or properly sized
            // For arm_interfaces::action::Gesture feedback structure (JointInfo[]),
            // we need to populate individual JointInfo messages.
            // This part requires more detailed knowledge of JointInfo struct.
            // Based on the image, JointInfo has name, number, angle_rad
            if (feedback->current_joint_angle.size() <= j) {
              feedback->current_joint_angle.resize(j + 1);
            }
            feedback->current_joint_angle.at(j).joint_idx.name = "joint_" +
              std::to_string(arm_joints_.at(j));                                                                      // Example name
            feedback->current_joint_angle.at(j).joint_idx.number = arm_joints_.at(j);
            feedback->current_joint_angle.at(j).angle_rad =
              state_msg_.motor_state().at(arm_joints_.at(j)).q();                                                      // Use actual state for feedback
          }
        }

        // Set weight (if needed for arm control mode, e.g., to enable torque control)
        // Assuming JointIndex::kNotUsedJoint is used for weight based on h1_2_arm_sdk_dds_example.cpp
        msg_.motor_cmd().at(JointIndex::kNotUsedJoint).q(1.0f); // Example: set weight to 1.0 during movement

        // Send dds msg
        arm_sdk_publisher_->Write(msg_);

        // Publish feedback
        if (goal_handle && feedback) {     // Check both pointers are valid
          goal_handle->publish_feedback(feedback);      // Pass the shared pointer directly
        }

        loop_rate.sleep();
      }
    } else if (action_type == ITS_ME_MOTION) {
      // Get current position from state_msg (ensure state_msg is fresh)
      // A more robust approach would wait for a state message if needed

      std::array<float, 15> current_jpos_des;

      // Get current position from state_msg (ensure state_msg is fresh)
      // A more robust approach would wait for a state message if needed
      for (int i = 0; i < arm_joints_.size(); ++i) {
        start_pos.at(i) = state_msg_.motor_state().at(arm_joints_.at(i)).q();
      }


      for (int i = 0; i <= num_time_steps; ++i) {
        if (goal_handle && goal_handle->is_canceling()) {
          RCLCPP_INFO(this->get_logger(), "Action canceled during movement");
          return;       // Exit movement loop
        }

        float phase = static_cast<float>(i) / num_time_steps;
        float smooth_phase = 0.5f - 0.5f * cos(Pi * phase);

        // sin 이징 interpolation
        for (int j = 0; j < arm_joints_.size(); ++j) {
          current_jpos_des.at(j) = start_pos.at(j) * (1.0f - smooth_phase) + init_pos_.at(j) *
            smooth_phase;

          // Set control commands
          msg_.motor_cmd().at(arm_joints_.at(j)).q(current_jpos_des.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).dq(dq_);       // Velocity command (usually 0 for position control)
          msg_.motor_cmd().at(arm_joints_.at(j)).kp(kp_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).kd(kd_array_.at(j));
          msg_.motor_cmd().at(arm_joints_.at(j)).tau(tau_ff_);       // Feedforward torque (usually 0)

          // Populate feedback (using actual state, not desired)
          if (feedback) {       // Use the passed feedback pointer
            // Ensure feedback->current_joint_angle has enough space or is resized
            // Assuming it's a dynamic container or properly sized
            // For arm_interfaces::action::Gesture feedback structure (JointInfo[]),
            // we need to populate individual JointInfo messages.
            // This part requires more detailed knowledge of JointInfo struct.
            // Based on the image, JointInfo has name, number, angle_rad
            if (feedback->current_joint_angle.size() <= j) {
              feedback->current_joint_angle.resize(j + 1);
            }
            feedback->current_joint_angle.at(j).joint_idx.name = "joint_" +
              std::to_string(arm_joints_.at(j));                                                                      // Example name
            feedback->current_joint_angle.at(j).joint_idx.number = arm_joints_.at(j);
            feedback->current_joint_angle.at(j).angle_rad =
              state_msg_.motor_state().at(arm_joints_.at(j)).q();                                                      // Use actual state for feedback
          }
        }

        // Set weight (if needed for arm control mode, e.g., to enable torque control)
        // Assuming JointIndex::kNotUsedJoint is used for weight based on h1_2_arm_sdk_dds_example.cpp
        msg_.motor_cmd().at(JointIndex::kNotUsedJoint).q(1.0f); // Example: set weight to 1.0 during movement

        // Send dds msg
        arm_sdk_publisher_->Write(msg_);

        // Publish feedback
        if (goal_handle && feedback) {     // Check both pointers are valid
          goal_handle->publish_feedback(feedback);      // Pass the shared pointer directly
        }

        loop_rate.sleep();
      }
    }


    RCLCPP_INFO(this->get_logger(), "Movement to target pose finished.");
  }

  // Initial arm setup (move to init_pos once)
  void initialize_arm()
  {
    std::cout << "Press ENTER to init arms ...";
    std::cin.get();

    RCLCPP_INFO(this->get_logger(), "Initializing arms...");

    // Get current joint position
    std::array<float, 15> current_jpos{};
    // Wait briefly for the first state message to arrive
    std::this_thread::sleep_for(std::chrono::milliseconds(500));     // Increased wait time
    for (int i = 0; i < arm_joints_.size(); ++i) {
      // Ensure state_msg_ is populated before reading
      if (state_msg_.motor_state().size() > arm_joints_.at(i)) {
        current_jpos.at(i) = state_msg_.motor_state().at(arm_joints_.at(i)).q();
      } else {
        RCLCPP_WARN(
          this->get_logger(),
          "Motor state not available for joint %d during initialization. Using 0.0.",
          arm_joints_.at(i));
        current_jpos.at(i) = 0.0f;          // Default to 0 if state not available
      }
    }
    RCLCPP_INFO(this->get_logger(), "Current joint position read.");


    float init_time = 3.0f;     // Duration for initial movement
    int init_time_steps = static_cast<int>(init_time / control_dt_);
    if (init_time_steps == 0) {init_time_steps = 1;}
    auto sleep_time = std::chrono::milliseconds(static_cast<int>(control_dt_ * 1000));

    for (int i = 0; i <= init_time_steps; ++i) {
      float weight = 1.0;       // Set weight to 1.0 during initialization
      msg_.motor_cmd().at(JointIndex::kNotUsedJoint).q(weight);
      float phase = 1.0f * i / init_time_steps;

      std::array<float, 15> current_jpos_des;
      for (int j = 0; j < arm_joints_.size(); ++j) {
        // Interpolate from current position to init_pos
        current_jpos_des.at(j) = current_jpos.at(j) * (1.0f - phase) + init_pos_.at(j) * phase;

        msg_.motor_cmd().at(arm_joints_.at(j)).q(current_jpos_des.at(j));
        msg_.motor_cmd().at(arm_joints_.at(j)).dq(dq_);
        msg_.motor_cmd().at(arm_joints_.at(j)).kp(kp_array_.at(j));
        msg_.motor_cmd().at(arm_joints_.at(j)).kd(kd_array_.at(j));
        msg_.motor_cmd().at(arm_joints_.at(j)).tau(tau_ff_);
      }

      arm_sdk_publisher_->Write(msg_);
      std::this_thread::sleep_for(sleep_time);
    }
    RCLCPP_INFO(this->get_logger(), "Arm initialization done.");
  }


  // Action Goal handling
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const arm_interfaces::action::Gesture::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received gesture action goal request: %d", goal->action);
    (void)uuid;

    if (is_action_active_) {
      RCLCPP_WARN(this->get_logger(), "Rejected goal: Another action is already active.");
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (goal->action == INIT_POS_MOTION || goal->action == WAVE_HAND_MOTION ||
      goal->action == FOLLOW_ME_MOTION || goal->action == ITS_ME_MOTION ||
      goal->action == INTRODUCE_MOTION)
    {
      return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    } else {
      RCLCPP_WARN(this->get_logger(), "Rejected goal: Invalid action type %d", goal->action);
      return rclcpp_action::GoalResponse::REJECT;
    }
  }

  // Action Cancel handling
  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<arm_interfaces::action::Gesture>>
    goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received request to cancel gesture action goal");
    (void)goal_handle;
    // Implement cancellation logic if needed (e.g., stop current movement)
    is_action_active_ = false;

    return rclcpp_action::CancelResponse::ACCEPT;
  }

  // Action Accepted handling and execution
  void handle_accepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<arm_interfaces::action::Gesture>>
    goal_handle)
  {
    // This function is called when the goal is accepted.
    // Start a new thread to execute the action so it doesn't block the main executor.
    std::thread{std::bind(&ArmComtrollerNode::execute, this, std::placeholders::_1),
      goal_handle}.detach();
  }


  // Action execution
  void execute(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<arm_interfaces::action::Gesture>>
    goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing gesture action");
    is_action_active_ = true;

    auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<arm_interfaces::action::Gesture::Feedback>();
    auto result = std::make_shared<arm_interfaces::action::Gesture::Result>();

    std::array<float, 15> target_pose;
    float move_duration = 3.0f;     // Duration for the action movement

    if (goal->action == INIT_POS_MOTION) {
      target_pose = init_pos_;       // Action 0: Move to init_pos
      RCLCPP_INFO(this->get_logger(), "Action 0: Moving to initial pose.");
    } else if (goal->action == WAVE_HAND_MOTION) {
      target_pose = wave_hand_init_pos_;       // Action 1: Move to target_pos (arms up)
      RCLCPP_INFO(this->get_logger(), "Action 1: Moving to wave hand initial pose.");
    } else if (goal->action == FOLLOW_ME_MOTION) {
      target_pose = follow_init_pos_;       // Action 2: Move to target_pos (arms up)
      RCLCPP_INFO(this->get_logger(), "Action 2: Moving to follow initial pose.");
    } else if (goal->action == ITS_ME_MOTION) {
      target_pose = its_me_init_pos_;       // Action 3: Move to target_pos (arms up)
      RCLCPP_INFO(this->get_logger(), "Action 3: Moving to its me initial pose.");
    } else if (goal->action == INTRODUCE_MOTION) {
      target_pose = introduce_init_pos_;       // Action 4: Move to target_pos (arms up)
      RCLCPP_INFO(this->get_logger(), "Action 4: Moving to introduce initial pose.");
    } else {
      // Should not happen due to handle_goal check, but as a fallback
      RCLCPP_ERROR(
        this->get_logger(), "Invalid action type received in execute: %d",
        goal->action);
      result->result = false;
      goal_handle->abort(result);       // Use the result object
      return;
    }
  }

  // Execute the movement, passing the feedback object
  move_arm_to_pose(goal->action, target_pose, move_duration, goal_handle, feedback);

  // Check if the goal was canceled
  if (goal_handle->is_canceling()) {
    result->result = false;         // Indicate failure due to cancellation
    goal_handle->canceled(result);         // Use the result object
    RCLCPP_INFO(this->get_logger(), "Gesture action canceled.");
  } else {
    // Action completed successfully
    result->result = true;         // Indicate success
    goal_handle->succeed(result);         // Use the result object
    RCLCPP_INFO(this->get_logger(), "Gesture action succeeded.");
  }

  is_action_active_ = false;
};

int main(int argc, char * argv[])
{
//   if (argc < 2) {
//     std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
//     return -1;
//   }

  rclcpp::init(argc, argv);

  // Pass the network interface to the node constructor
  auto node = std::make_shared<ArmComtrollerNode>("eno1");

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
