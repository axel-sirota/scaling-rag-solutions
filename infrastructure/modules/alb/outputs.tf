output "target_group_arn" {
  value = aws_lb_target_group.rag_tg.arn
}

output "alb_dns_name" {
  value = aws_lb.rag_alb.dns_name
}
