output "alb_dns_name" {
  description = "The DNS name of the ALB endpoint"
  value       = module.alb.alb_dns_name
}
