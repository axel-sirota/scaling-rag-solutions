output "vpc_id" {
  value = aws_vpc.rag_vpc.id
}

output "public_subnets" {
  value = [
    aws_subnet.public_subnet.id,
    aws_subnet.public_subnet_b.id
  ]
}

output "ecs_sg_id" {
  value = aws_security_group.ecs_sg.id
}
