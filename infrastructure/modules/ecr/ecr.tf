resource "aws_ecr_repository" "rag_ecr" {
  name                 = "scaled-rag-repo"
  image_tag_mutability = "MUTABLE"
  lifecycle {
    prevent_destroy = true
  }
}
