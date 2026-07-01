# Policy and Testing Examples

OPA policies, Sentinel rules, and infrastructure testing patterns.

## OPA (Open Policy Agent)

### Required Tags Policy
```rego
# required_tags.rego
package terraform.tags

import future.keywords.if
import future.keywords.in

deny[msg] if {
  resource := input.resource_changes[_]
  resource.mode == "managed"
  not resource.change.after.tags
  msg := sprintf("Resource %s missing required tags", [resource.address])
}

deny[msg] if {
  resource := input.resource_changes[_]
  required_tags := {"Environment", "Project", "Owner"}
  missing := required_tags - object.keys(resource.change.after.tags)
  count(missing) > 0
  msg := sprintf("Resource %s missing tags: %v", [resource.address, missing])
}
```

### Encryption Policy
```rego
# encryption_required.rego
package terraform.encryption

deny[msg] if {
  resource := input.resource_changes[_]
  resource.type == "aws_s3_bucket"
  not resource.change.after.server_side_encryption_configuration
  msg := sprintf("S3 bucket %s must have encryption enabled", [resource.address])
}
```

### Cost Control Policy
```rego
# cost_control.rego
package terraform.cost

deny[msg] if {
  resource := input.resource_changes[_]
  resource.type == "aws_instance"
  instance_type := resource.change.after.instance_type
  not startswith(instance_type, "t3.")
  msg := sprintf("EC2 instance %s uses expensive type %s. Use t3.* for dev.", [resource.address, instance_type])
}
```

## Sentinel (Terraform Cloud)

### Require Encryption
```hcl
# require_encryption.sentinel
import "tfplan"

main = rule {
  all tfplan.resources.aws_s3_bucket as _, buckets {
    all buckets as _, bucket {
      bucket.applied.server_side_encryption_configuration is not null
    }
  }
}
```

### Restrict Instance Types
```hcl
# restrict_instance_types.sentinel
import "tfplan"

allowed_types = ["t3.micro", "t3.small", "t3.medium"]

main = rule {
  all tfplan.resources.aws_instance as _, instances {
    all instances as _, instance {
      instance.applied.instance_type in allowed_types
    }
  }
}
```

## Terratest (Go)

### VPC Module Test
```go
// vpc_test.go
package test

import (
  "testing"
  "github.com/gruntwork-io/terratest/modules/terraform"
  "github.com/stretchr/testify/assert"
)

func TestVpcModule(t *testing.T) {
  terraformOptions := &terraform.Options{
    TerraformDir: "../modules/vpc",
    Vars: map[string]interface{}{
      "name":               "test-vpc",
      "cidr_block":          "10.0.0.0/16",
      "availability_zones":  []string{"us-east-1a", "us-east-1b"},
    },
  }

  defer terraform.Destroy(t, terraformOptions)
  terraform.InitAndApply(t, terraformOptions)

  vpcId := terraform.Output(t, terraformOptions, "vpc_id")
  assert.NotEmpty(t, vpcId)

  privateSubnets := terraform.OutputList(t, terraformOptions, "private_subnet_ids")
  assert.Equal(t, 2, len(privateSubnets))
}
```

### Database Module Test
```go
// database_test.go
package test

import (
  "testing"
  "github.com/gruntwork-io/terratest/modules/terraform"
  "github.com/stretchr/testify/assert"
)

func TestDatabaseModule(t *testing.T) {
  terraformOptions := &terraform.Options{
    TerraformDir: "../modules/database",
    Vars: map[string]interface{}{
      "identifier":     "test-db",
      "engine":         "postgres",
      "instance_class": "db.t3.micro",
    },
  }

  defer terraform.Destroy(t, terraformOptions)
  terraform.InitAndApply(t, terraformOptions)

  endpoint := terraform.Output(t, terraformOptions, "endpoint")
  assert.Contains(t, endpoint, "rds.amazonaws.com")
}
```

## Kitchen-Terraform (Ruby)

### VPC Controls
```ruby
# controls/vpc.rb
control 'vpc-exists' do
  describe aws_vpc('vpc-12345678') do
    it { should exist }
    its('cidr_block') { should eq '10.0.0.0/16' }
  end
end

control 'subnets-exist' do
  describe aws_subnets do
    its('subnet_ids.count') { should be >= 2 }
  end
end
```

### Security Group Controls
```ruby
# controls/security_group.rb
control 'no-ssh-from-internet' do
  aws_security_groups.group_ids.each do |sg_id|
    describe aws_security_group(sg_id) do
      it { should_not allow_in(port: 22, ipv4_range: '0.0.0.0/0') }
    end
  end
end
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/terraform.yml
name: Terraform

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        
      - name: Terraform Format
        run: terraform fmt -check -recursive
        
      - name: Terraform Validate
        run: |
          terraform init -backend=false
          terraform validate
          
      - name: Run TFLint
        uses: terraform-linters/setup-tflint@v4
        with:
          tflint_version: latest
      - run: tflint --init && tflint
        
      - name: Run Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: .
          framework: terraform
          
      - name: Run OPA Tests
        run: |
          terraform plan -out=tfplan
          terraform show -json tfplan > tfplan.json
          opa test policies/ --verbose
          opa eval --data policies/required_tags.rego --input tfplan.json "data.terraform.tags.deny"
```

### Validation Script
```bash
#!/bin/bash
# validate.sh

set -e

echo "Running Terraform validation..."

# Format check
echo "  → Checking format..."
terraform fmt -check -recursive

# Initialize
echo "  → Initializing..."
terraform init -backend=false

# Validate
echo "  → Validating..."
terraform validate

# Security scan with Checkov
echo "  → Running Checkov..."
checkov -d . --framework terraform --quiet

# Lint with TFLint
echo "  → Running TFLint..."
tflint --init
tflint

# Plan and OPA check (if policies exist)
if [ -d "policies" ]; then
  echo "  → Running OPA policy checks..."
  terraform plan -out=tfplan
  terraform show -json tfplan > tfplan.json
  opa test policies/
fi

echo "All validation passed!"
```
