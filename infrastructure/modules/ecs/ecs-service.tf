resource "aws_ecs_cluster" "rag_cluster" {
  name = "scaled-rag-cluster"
}

resource "aws_ecs_service" "rag_service" {
  name            = "rag-service"
  cluster         = aws_ecs_cluster.rag_cluster.id
  task_definition = aws_ecs_task_definition.rag_task.arn
  desired_count   = 2
  launch_type     = "EC2"

  load_balancer {
    target_group_arn = var.target_group_arn
    container_name   = "rag-container"
    container_port   = 8000
  }
  health_check_grace_period_seconds = 300
}
