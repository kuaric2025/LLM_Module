#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <urdf/model.h>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/frames.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>  // With joint limits

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "../../../install/ik_solver_pkg/include/ik_solver_pkg/ik_solver_pkg/srv/set_target_pose.hpp" 
// #include "../../install/ik_solver_pkg/include/ik_solver_pkg/ik_solver_pkg/srv/set_target_pose.hpp" 

class IKNode : public rclcpp::Node
{
public:
    IKNode() : Node("ik_solver_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        RCLCPP_INFO(this->get_logger(), "🚀 IKNode constructor started");
        std::string urdf_path = "/home/kucarst3-dlws/Documents/isaam-sim/env_isaaclab/lib/python3.10/site-packages/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10e/ur10e.urdf";
        urdf::Model robot_model;
        if (!robot_model.initFile(urdf_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF file");
            return;
        }

        KDL::Tree kdl_tree;
        if (!kdl_parser::treeFromUrdfModel(robot_model, kdl_tree)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to construct KDL tree");
            return;
        }

        std::string base_link = "base_link";
        std::string ee_link = "tool0";

        if (!kdl_tree.getChain(base_link, ee_link, chain_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to extract KDL chain");
            return;
        }

        n_joints_ = chain_.getNrOfJoints();
        fk_solver_ = std::make_shared<KDL::ChainFkSolverPos_recursive>(chain_);
        ik_solver_vel_ = std::make_shared<KDL::ChainIkSolverVel_pinv>(chain_);

        // Setup joint limits
        q_min_.resize(n_joints_);
        q_max_.resize(n_joints_);

        unsigned int j = 0;
        for (const auto& joint : robot_model.joints_) {
            if (joint.second->type != urdf::Joint::FIXED) {
                q_min_(j) = joint.second->limits ? joint.second->limits->lower : -M_PI;
                q_max_(j) = joint.second->limits ? joint.second->limits->upper : M_PI;
                j++;
                if (j >= n_joints_) break;
            }
        }

        ik_solver_ = std::make_shared<KDL::ChainIkSolverPos_NR_JL>(chain_, q_min_, q_max_, *fk_solver_, *ik_solver_vel_, 200, 1e-6);

        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/isaac_joint_commands", 10);
        // timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&IKNode::handleIKRequest, this));//tryMove

        joint_names_ = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"};

        last_joint_config_.resize(n_joints_);
        std::vector<double> initial_guess = {0.0, -1.4, 1.4, -1.6, -1.6, 0.0};
        for (unsigned int i = 0; i < n_joints_; ++i)
            last_joint_config_(i) = initial_guess[i];

        has_last_config_ = true;

        ik_service_ = this->create_service<ik_solver_pkg::srv::SetTargetPose>(
        "set_target_pose",
        std::bind(&IKNode::handleIKRequest, this, std::placeholders::_1, std::placeholders::_2)
);

    }

private:
    void handleIKRequest(
        const std::shared_ptr<ik_solver_pkg::srv::SetTargetPose::Request> request,
        std::shared_ptr<ik_solver_pkg::srv::SetTargetPose::Response> response)
    {
        const auto &pose = request->target_pose;

        KDL::Frame target_frame(
            KDL::Rotation::Quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            KDL::Vector(pose.position.x, pose.position.y, pose.position.z));

        // --- Apply offset from TCP to actual grasp point ---
        KDL::Frame tcp_to_grasp_offset(
            KDL::Rotation::Quaternion(0.0, 0.0, 0.0, 1.0),   // Adjust this rotation as needed
            KDL::Vector(0.0, 0.0, -0.26)      // Adjust this translation as needed -0.24
        );

        // New frame: where TCP must be so that the gripper aligns with the target
        KDL::Frame target_frame_with_offset = tcp_to_grasp_offset * target_frame;



        KDL::JntArray q_init = last_joint_config_;
        KDL::JntArray q_result(n_joints_);

        if (ik_solver_->CartToJnt(q_init, target_frame_with_offset, q_result) >= 0) {
            sensor_msgs::msg::JointState joint_msg;
            joint_msg.header.stamp = this->get_clock()->now();
            joint_msg.name = joint_names_;
            for (unsigned int i = 0; i < n_joints_; ++i)
                joint_msg.position.push_back(q_result(i));

            // joint_pub_->publish(joint_msg);
            // last_joint_config_ = q_result;

            // Only move robot if requested!
            if (request->execute) {
                joint_pub_->publish(joint_msg);
                last_joint_config_ = q_result;
                response->message = "IK solution found and sent.";
            } else {
                response->message = "IK solution found (not sent).";
            }

            // Always return the solution in the response
            response->joint_state = joint_msg;
            response->success = true;

            RCLCPP_INFO(this->get_logger(), "✅ IK succeeded");
        } else {
            response->success = false;
            response->message = "IK solver failed.";
            RCLCPP_ERROR(this->get_logger(), "❌ IK failed");
        }
    }

    void tryMove()
    {
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_.lookupTransform("base_link", "tool0", tf2::TimePointZero);
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            return;
        }

        const auto &t = transform.transform.translation;
        const auto &r = transform.transform.rotation;

        KDL::Frame current_frame(
            KDL::Rotation::Quaternion(r.x, r.y, r.z, r.w),
            KDL::Vector(t.x, t.y, t.z));

        KDL::Frame target_frame = current_frame;
        target_frame.p.data[0] += 0.2;  // move 20 cm in X

        KDL::JntArray q_init = last_joint_config_;
        KDL::JntArray q_result(n_joints_);

        if (ik_solver_->CartToJnt(q_init, target_frame, q_result) >= 0) {
            RCLCPP_INFO(this->get_logger(), "✅ IK tryMove succeeded");

            sensor_msgs::msg::JointState joint_msg;
            joint_msg.header.stamp = this->get_clock()->now();
            joint_msg.name = joint_names_;
            for (unsigned int i = 0; i < n_joints_; ++i)
                joint_msg.position.push_back(q_result(i));

            joint_pub_->publish(joint_msg);
            last_joint_config_ = q_result;
            has_last_config_ = true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "❌ IK tryMove failed");
        }
    }

    // IK core components
    KDL::Chain chain_;
    std::shared_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::shared_ptr<KDL::ChainIkSolverVel_pinv> ik_solver_vel_;
    std::shared_ptr<KDL::ChainIkSolverPos_NR_JL> ik_solver_;

    KDL::JntArray q_min_, q_max_;
    KDL::JntArray last_joint_config_;
    bool has_last_config_ = false;

    size_t n_joints_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Service<ik_solver_pkg::srv::SetTargetPose>::SharedPtr ik_service_;
    rclcpp::TimerBase::SharedPtr timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    std::vector<std::string> joint_names_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IKNode>());
    rclcpp::shutdown();
    return 0;
}