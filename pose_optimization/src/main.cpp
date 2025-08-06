#include "pose_optimization/optimizer.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseOptimizer>());
    rclcpp::shutdown();
    return 0;
}
