provider "aws" {
  region  = "us-east-1"
  profile = "default"
}

module "network" {
  source = "./modules/network"
}

module "alb" {
  source         = "./modules/alb"
  vpc_id         = module.network.vpc_id
  public_subnets = module.network.public_subnets
  ecs_sg_id      = module.network.ecs_sg_id
}

module "ecs" {
  source           = "./modules/ecs"
  # ECS module expects one public subnet – we choose the first from the list.
  public_subnet      = module.network.public_subnets[0]
  ecs_sg_id          = module.network.ecs_sg_id
  target_group_arn   = module.alb.target_group_arn
  public_subnet_id   = module.network.public_subnets[0]
  public_subnets    = module.network.public_subnets
}

module "ecr" {
  source = "./modules/ecr"
}
