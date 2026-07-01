# Multi-Cloud Resource Examples

Detailed HCL examples for AWS, GCP, Azure, and Oracle Cloud.

## AWS Examples

### ECS Fargate Service
```hcl
resource "aws_ecs_service" "api" {
  name            = "${local.prefix}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_min_instances
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
}
```

### OIDC for GitHub Actions
```hcl
resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4e98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github_actions" {
  name = "${local.prefix}-github-actions"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = { "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com" }
        StringLike = { "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*" }
      }
    }]
  })
}
```

### Secrets Manager
```hcl
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "${local.prefix}/db-password"
}
```

## GCP Examples

### Cloud Run Service
```hcl
resource "google_cloud_run_v2_service" "api" {
  name     = "${local.prefix}-api"
  location = var.region
  project  = var.project_id

  template {
    service_account = google_service_account.api.email
    
    scaling {
      min_instances = var.api_min_instances
      max_instances = var.api_max_instances
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${local.prefix}/api:${var.image_tag}"
      
      resources {
        limits = { cpu = "2", memory = "1Gi" }
        cpu_idle = true
      }

      ports { container_port = 8000 }
    }

    vpc_access {
      connector = google_vpc_access_connector.main.id
      egress    = "ALL_TRAFFIC"
    }
  }
}
```

### Workload Identity Federation
```hcl
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions Pool"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}
```

### Secret Manager
```hcl
data "google_secret_manager_secret_version" "db_password" {
  secret  = "db-password"
  version = "latest"
}
```

## Azure Examples

### Container Apps
```hcl
resource "azurerm_container_app" "api" {
  name                         = "${local.prefix}-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    min_replicas = var.api_min_instances
    max_replicas = var.api_max_instances

    container {
      name   = "api"
      image  = "${azurerm_container_registry.main.login_server}/${local.prefix}/api:${var.image_tag}"
      cpu    = 1.0
      memory = "2Gi"

      env {
        name  = "DATABASE_URL"
        value = "postgresql://..."
      }
    }
  }
}
```

### Federated Credentials
```hcl
resource "azuread_application_federated_identity_credential" "github" {
  application_object_id = azuread_application.github.object_id
  display_name          = "github-actions"
  description           = "GitHub Actions OIDC"
  audiences             = ["api://AzureADTokenExchange"]
  issuer                = "https://token.actions.githubusercontent.com"
  subject               = "repo:${var.github_repo}:ref:refs/heads/main"
}
```

### Key Vault
```hcl
data "azurerm_key_vault_secret" "db_password" {
  name         = "db-password"
  key_vault_id = azurerm_key_vault.main.id
}
```

## Oracle Cloud Examples

### Container Instances
```hcl
resource "oci_container_instances_container_instance" "api" {
  compartment_id = var.compartment_id
  display_name   = "${local.prefix}-api"
  
  containers {
    image_url = "${oci_artifacts_container_repository.api.repository_url}:${var.image_tag}"
    display_name = "api"
    
    environment_variables = {
      "DATABASE_URL" = "postgresql://..."
    }
  }
}
```

### Vault
```hcl
data "oci_secrets_secretbundle" "db_password" {
  secret_id = oci_vault_secret.db_password.id
}
```
