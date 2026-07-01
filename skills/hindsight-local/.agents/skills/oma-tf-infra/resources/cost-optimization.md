# Cost Optimization Examples

Terraform patterns for optimizing cloud infrastructure costs across AWS, GCP, and Azure.

## Environment-Based Resource Sizing

Dynamically size resources based on environment to reduce non-production costs.

```hcl
# locals.tf
locals {
  # Instance sizing by environment
  instance_sizes = {
    production  = "t3.large"
    staging     = "t3.medium"
    development = "t3.micro"
  }

  # Database sizing by environment  
  db_instance_classes = {
    production  = "db.r5.xlarge"
    staging     = "db.t3.medium"
    development = "db.t3.micro"
  }

  # Enable HA only in production
  enable_multi_az = var.environment == "production" ? true : false
  enable_backups  = var.environment == "production" ? true : false
}

# Compute resource with environment-based sizing
resource "aws_instance" "app" {
  instance_type = local.instance_sizes[var.environment]
  # ...
}

# Database with environment-based sizing
resource "aws_db_instance" "main" {
  instance_class = local.db_instance_classes[var.environment]
  multi_az       = local.enable_multi_az
  # ...
}
```

## Reserved Instances & Savings Plans

### AWS Reserved Instances

```hcl
# Data source for Reserved Instance offerings
data "aws_ec2_instance_type_offerings" "available" {
  filter {
    name   = "instance-type"
    values = ["t3.medium", "t3.large"]
  }
}

# Note: Reserved Instances are purchased via Console or API, not Terraform
# Use tags to track RI-eligible resources
resource "aws_instance" "app" {
  instance_type = "t3.medium"
  
  tags = {
    Name        = "app-server"
    RIEligible  = "true"
    Environment = "production"
  }
}
```

### GCP Committed Use Discounts (CUDs)

```hcl
# Committed use discount for predictable workloads
resource "google_compute_resource_policy" "cud" {
  name   = "cud-policy"
  region = var.region

  group_placement_policy {
    availability_domain_count = 1
  }
}

# Note: CUDs are purchased in console, tag resources for tracking
resource "google_compute_instance" "app" {
  machine_type = "n2-standard-4"
  
  labels = {
    cud_eligible = "true"
  }
}
```

### Azure Reserved VM Instances

```hcl
# Note: Reserved Instances are purchased separately
# Tag resources for RI tracking
resource "azurerm_linux_virtual_machine" "app" {
  size = "Standard_D4s_v3"
  
  tags = {
    RIEligible = "true"
  }
}
```

## Spot/Preemptible Instances

For fault-tolerant workloads, use spot instances for up to 90% savings.

### AWS Spot Instances

```hcl
resource "aws_launch_template" "spot" {
  name_prefix   = "spot-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = "t3.medium"

  # Spot instance configuration
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "0.05"  # Maximum spot price willing to pay
      spot_instance_type = "one-time"
    }
  }
}

resource "aws_autoscaling_group" "spot" {
  name                = "spot-asg"
  vpc_zone_identifier = var.private_subnets
  target_group_arns   = [aws_lb_target_group.app.arn]
  health_check_type   = "ELB"

  min_size         = 1
  max_size         = 10
  desired_capacity = 3

  launch_template {
    id      = aws_launch_template.spot.id
    version = "$Latest"
  }

  # Mixed instances policy for spot diversification
  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.spot.id
      }

      override {
        instance_type = "t3.medium"
      }
      override {
        instance_type = "t3a.medium"
      }
      override {
        instance_type = "m5.medium"
      }
    }

    instances_distribution {
      on_demand_base_capacity                  = 1
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "capacity-optimized"
    }
  }
}
```

### GCP Preemptible VMs

```hcl
resource "google_compute_instance" "worker" {
  name         = "worker"
  machine_type = "n2-standard-4"
  zone         = var.zone

  scheduling {
    preemptible       = true
    automatic_restart = false
  }

  # Spot VM (newer term for preemptible)
  resource_policies = [google_compute_resource_policy.spot.id]
}

resource "google_compute_resource_policy" "spot" {
  name   = "spot-policy"
  region = var.region

  instance_schedule_policy {
    vm_start_schedule {
      schedule = "0 8 * * MON-FRI"
    }
    vm_stop_schedule {
      schedule = "0 20 * * MON-FRI"
    }
    time_zone = "America/New_York"
  }
}
```

### Azure Spot VMs

```hcl
resource "azurerm_linux_virtual_machine" "spot" {
  name                = "spot-vm"
  resource_group_name = var.resource_group_name
  location            = var.location
  size                = "Standard_D4s_v3"
  priority            = "Spot"
  eviction_policy     = "Deallocate"  # or "Delete"

  # Only create if spot price is acceptable
  max_bid_price = "0.1"
}
```

## Autoscaling Schedules

Scale down during non-business hours to reduce costs.

### AWS Autoscaling Schedule

```hcl
# Scale down in the evening
resource "aws_autoscaling_schedule" "scale_down_evening" {
  scheduled_action_name  = "scale-down-evening"
  min_size               = 1
  max_size               = 1
  desired_capacity       = 1
  recurrence             = "0 20 * * MON-FRI"  # 8 PM weekdays
  autoscaling_group_name = aws_autoscaling_group.app.name
}

# Scale up in the morning
resource "aws_autoscaling_schedule" "scale_up_morning" {
  scheduled_action_name  = "scale-up-morning"
  min_size               = 3
  max_size               = 10
  desired_capacity       = 3
  recurrence             = "0 7 * * MON-FRI"   # 7 AM weekdays
  autoscaling_group_name = aws_autoscaling_group.app.name
}

# Weekend shutdown
resource "aws_autoscaling_schedule" "weekend_shutdown" {
  scheduled_action_name  = "weekend-shutdown"
  min_size               = 0
  max_size               = 0
  desired_capacity       = 0
  recurrence             = "0 18 * * FRI"      # 6 PM Friday
  autoscaling_group_name = aws_autoscaling_group.app.name
}
```

### GCP Autoscaler Schedules

```hcl
resource "google_compute_autoscaler" "app" {
  name   = "app-autoscaler"
  zone   = var.zone
  target = google_compute_instance_group_manager.app.id

  autoscaling_policy {
    min_replicas    = 1
    max_replicas    = 10
    cooldown_period = 60

    cpu_utilization {
      target = 0.6
    }

    scaling_schedules {
      name               = "scale-down-evening"
      description        = "Scale down in the evening"
      min_required_replicas = 1
      schedule           = "0 20 * * 1-5"  # 8 PM weekdays
      time_zone          = "America/New_York"
      duration_sec       = 39600  # 11 hours
    }

    scaling_schedules {
      name               = "scale-up-morning"
      description        = "Scale up in the morning"
      min_required_replicas = 3
      schedule           = "0 7 * * 1-5"   # 7 AM weekdays
      time_zone          = "America/New_York"
      duration_sec       = 46800  # 13 hours
    }
  }
}
```

## Storage Lifecycle Rules

Transition data to cheaper storage classes based on age.

### AWS S3 Lifecycle Rules

```hcl
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    # Transition to Infrequent Access after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Transition to Glacier after 90 days
    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    # Delete after 1 year
    expiration {
      days = 365
    }

    # Clean up incomplete multipart uploads
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  # Rule for temporary data
  rule {
    id     = "delete-temp-data"
    status = "Enabled"

    filter {
      prefix = "temp/"
    }

    expiration {
      days = 7
    }
  }
}

# Enable Intelligent-Tiering for automatic cost optimization
resource "aws_s3_bucket_intelligent_tiering_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  name   = "EntireBucket"

  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 90
  }

  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 180
  }
}
```

### GCP Storage Lifecycle Rules

```hcl
resource "google_storage_bucket" "data" {
  name          = "${var.project_id}-data"
  location      = var.region
  storage_class = "STANDARD"

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
}
```

### Azure Blob Storage Lifecycle

```hcl
resource "azurerm_storage_management_policy" "data" {
  storage_account_id = azurerm_storage_account.data.id

  rule {
    name    = "transition-to-cool"
    enabled = true
    filters {
      prefix_match = ["logs/"]
      blob_types   = ["blockBlob"]
    }
    actions {
      base_blob {
        tier_to_cool_after_days_since_modification_greater_than    = 30
        tier_to_archive_after_days_since_modification_greater_than = 90
        delete_after_days_since_modification_greater_than          = 365
      }
      snapshot {
        delete_after_days_since_creation_greater_than = 30
      }
    }
  }
}
```

## Cost Allocation Tags

Tag all resources for cost tracking and chargeback.

### AWS Cost Allocation Tags

```hcl
locals {
  common_tags = {
    Environment     = var.environment
    Project         = var.project_name
    Owner           = var.team_email
    CostCenter      = var.cost_center
    ManagedBy       = "terraform"
  }
}

# Apply tags to all resources
resource "aws_instance" "app" {
  ami           = data.aws_ami.amazon_linux.id
  instance_type = "t3.medium"

  tags = local.common_tags
}

resource "aws_db_instance" "main" {
  identifier     = "${var.project_name}-db"
  instance_class = "db.t3.medium"

  tags = local.common_tags
}

# Activate cost allocation tags (run once per account)
resource "aws_ce_cost_allocation_tag" "environment" {
  tag_key = "Environment"
  status  = "Active"
}

resource "aws_ce_cost_allocation_tag" "project" {
  tag_key = "Project"
  status  = "Active"
}
```

### GCP Labels

```hcl
locals {
  common_labels = {
    environment = var.environment
    project     = var.project_name
    owner       = var.team_email
    cost-center = var.cost_center
    managed-by  = "terraform"
  }
}

resource "google_compute_instance" "app" {
  name         = "app-server"
  machine_type = "n2-standard-4"

  labels = local.common_labels
}

# Export billing data to BigQuery for analysis
resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account
  display_name    = "Monthly Budget"

  amount {
    specified_amount {
      currency_code = "USD"
      units         = "1000"
    }
  }

  threshold_rules {
    threshold_percent = 50
  }
  threshold_rules {
    threshold_percent = 90
  }
}
```

### Azure Tags

```hcl
locals {
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    Owner       = var.team_email
    CostCenter  = var.cost_center
    ManagedBy   = "terraform"
  }
}

resource "azurerm_linux_virtual_machine" "app" {
  name                = "app-vm"
  resource_group_name = var.resource_group_name
  location            = var.location
  size                = "Standard_D4s_v3"

  tags = local.common_tags
}

# Cost Management Budget
resource "azurerm_consumption_budget_resource_group" "monthly" {
  name              = "monthly-budget"
  resource_group_id = var.resource_group_id
  amount            = 1000
  time_grain        = "Monthly"

  time_period {
    start_date = "2024-01-01T00:00:00Z"
    end_date   = "2024-12-31T00:00:00Z"
  }

  notification {
    enabled        = true
    threshold      = 90
    operator       = "GreaterThan"
    contact_emails = [var.team_email]
  }
}
```

## Best Practices Summary

1. **Use environment-based sizing** - Smaller instances for dev/staging
2. **Reserved capacity for production** - 1-3 year commitments for predictable workloads
3. **Spot instances for fault-tolerant workloads** - Up to 90% savings
4. **Scheduled scaling** - Scale down during off-hours
5. **Storage lifecycle policies** - Automate transitions to cheaper tiers
6. **Cost allocation tags** - Track and optimize spending
7. **Right-sizing** - Use the smallest instance that meets performance needs
8. **Auto-shutdown** - Stop dev/test resources when not in use
