resource "aws_ecs_task_definition" "rag_task" {
  family                   = "rag-task"
  requires_compatibilities = ["EC2"]
  memory                   = "32768"
  cpu                      = "4096"
  network_mode             = "bridge"
  
  container_definitions = jsonencode([
    {
      name         = "rag-container"
      image        = "253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:latest"
      essential    = true
      memory       = 32768
      cpu          = 4096
      resourceRequirements = [
        {
          type  = "GPU"
          value = "1"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options   = {
          "awslogs-group"         = "/ecs/rag-service"
          "awslogs-region"        = "us-east-1"
          "awslogs-stream-prefix" = "rag-container"
        }
      }
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
    }
  ])
}
