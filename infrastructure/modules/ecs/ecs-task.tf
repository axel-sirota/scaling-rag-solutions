resource "aws_ecs_task_definition" "rag_task" {
  family                   = "rag-task"
  requires_compatibilities = ["EC2"]
  memory                   = "8192"
  cpu                      = "4096"

  container_definitions = jsonencode([
    {
      name         = "rag-container"
      image        = "253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:latest"
      essential    = true
      memory       = 8192
      cpu          = 4096
      resourceRequirements = [
        {
          type  = "GPU"
          value = "2"
        }
      ]
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
