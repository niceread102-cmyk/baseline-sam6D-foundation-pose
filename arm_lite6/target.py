import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point


class TargetPublisher(Node):
    def __init__(self):
        super().__init__('target_publisher')
        self.pub = self.create_publisher(Point, '/target_pos', 10)


def main(args=None):
    rclpy.init(args=args)
    node = TargetPublisher()
    try:
        while rclpy.ok():
            raw = input("Enter target x y z (metres): ").strip()
            x, y, z = map(float, raw.split())
            msg = Point(x=x, y=y, z=z)
            node.pub.publish(msg)
            node.get_logger().info(f"Published [{x:.3f}, {y:.3f}, {z:.3f}]")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
