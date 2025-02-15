variable "public_subnet" {
  description = "A single public subnet for ECS service"
  type        = string
}

variable "target_group_arn" {
  description = "The ARN of the target group from ALB"
  type        = string
}

variable "ecs_sg_id" {
  description = "Security Group ID for ECS instances"
  type        = string
}

variable "public_subnet_id" {
  description = "The public subnet ID for the autoscaling group"
  type        = string
}

variable "public_subnets" {
  description = "List of public subnet IDs for the ASG"
  type        = list(string)
}