data "aws_ssm_parameter" "ecs_gpu_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id"
}

resource "aws_launch_template" "ecs_gpu" {
  name          = "ecs-gpu-template"
  image_id      = data.aws_ssm_parameter.ecs_gpu_ami.value
  instance_type = "g4dn.2xlarge"

  user_data = base64encode(<<EOF
#!/bin/bash
echo "ECS_CLUSTER=scaled-rag-cluster" >> /etc/ecs/ecs.config
echo "ECS_ENABLE_GPU_SUPPORT=true" >> /etc/ecs/ecs.config
EOF
  )

  iam_instance_profile {
    name = aws_iam_instance_profile.ecs_instance_profile.name
  }

  key_name = "ecs-rag"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 150
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
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }
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

resource "aws_cloudwatch_log_group" "rag_logs" {
  name              = "/ecs/rag-service"
  retention_in_days = 7
}
