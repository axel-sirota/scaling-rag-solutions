variable "vpc_id" {
  description = "The VPC ID where the ALB will be deployed"
  type        = string
}

variable "public_subnets" {
  description = "List of public subnets for the ALB"
  type        = list(string)
}

variable "ecs_sg_id" {
  description = "Security Group ID to attach to the ALB"
  type        = string
}
