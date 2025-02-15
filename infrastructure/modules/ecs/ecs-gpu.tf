resource "aws_launch_template" "ecs_gpu" {
  name          = "ecs-gpu-template"
  image_id      = "ami-02ca4fbacf5f85ce5"  # ECS-optimized Amazon Linux 2 AMI with OSS NVIDIA driver & PyTorch
  instance_type = "g4dn.2xlarge"

  iam_instance_profile {
    name = aws_iam_instance_profile.ecs_instance_profile.name
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 100
      volume_type = "gp2"
    }
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [var.ecs_sg_id]
  }
  instance_market_options {
    market_type = "spot"
    spot_options {
      # Use "one-time" or "persistent" depending on your needs.
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

  user_data = base64encode(<<EOF
#!/bin/bash
# Configure the ECS agent to join your cluster
echo "ECS_CLUSTER=scaled-rag-cluster" >> /etc/ecs/ecs.config
EOF
  )
}

resource "aws_autoscaling_group" "ecs_asg" {
  desired_capacity    = 1
  max_size            = 2
  min_size            = 1
  vpc_zone_identifier = var.public_subnets


  launch_template {
    id      = aws_launch_template.ecs_gpu.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "ecs-gpu-instance"
    propagate_at_launch = true
  }
}
