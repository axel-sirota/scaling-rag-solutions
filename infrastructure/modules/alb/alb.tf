############################################
# ALB with Access Logs
############################################
resource "aws_lb" "rag_alb" {
  name               = "rag-load-balancer"
  load_balancer_type = "application"
  security_groups    = [var.ecs_sg_id]
  subnets            = var.public_subnets
  internal           = false
  idle_timeout = 600
  
}

resource "aws_lb_target_group" "rag_tg" {
  name     = "rag-target-group"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = var.vpc_id
  health_check {
    path                = "/"
    port                = "traffic-port"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 45
    timeout             = 30
    healthy_threshold   = 3
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.rag_alb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.rag_tg.arn
  }
}
