#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy Hindsight Agent API to Azure using Bicep

.DESCRIPTION
    This script deploys the Hindsight Agent API infrastructure using Bicep
    and builds/pushes the container image to ACR.

.PARAMETER ResourceGroup
    The Azure resource group name (default: hindsight-rg)

.PARAMETER Location
    The Azure region (default: centralus)

.EXAMPLE
    .\deploy-bicep.ps1
    .\deploy-bicep.ps1 -ResourceGroup my-rg -Location eastus
#>

param(
    [string]$ResourceGroup = "hindsight-rg",
    [string]$Location = "centralus",
    [string]$ImageTag = "latest",
    [string]$AiResourceGroup = "jacob-1216-resource",
    [string]$AiResourceName = "jacob-1216-resource"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Deploying Hindsight Agent API with Bicep" -ForegroundColor Cyan
Write-Host "   Resource Group: $ResourceGroup"
Write-Host "   Location: $Location"

# Check Azure CLI login
Write-Host "`nChecking Azure CLI authentication..." -ForegroundColor Yellow
$accountJson = az account show 2>$null
if (-not $accountJson) {
    Write-Host "Not logged in to Azure CLI. Run 'az login' first." -ForegroundColor Red
    exit 1
}
$account = $accountJson | ConvertFrom-Json
Write-Host "   Logged in as: $($account.user.name)"
Write-Host "   Subscription: $($account.name)"

# Create resource group if needed
Write-Host "`nEnsuring resource group exists..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location --output none 2>$null

# Deploy Bicep template
Write-Host "`nDeploying infrastructure with Bicep..." -ForegroundColor Yellow
$deploymentJson = az deployment group create --resource-group $ResourceGroup --template-file "$ScriptDir/infra/agent-api.bicep" --parameters location=$Location imageTag=$ImageTag aiProjectResourceGroup=$AiResourceGroup aiResourceName=$AiResourceName --query "properties.outputs" --output json

if (-not $deploymentJson) {
    Write-Host "Bicep deployment failed" -ForegroundColor Red
    exit 1
}

$deploymentOutput = $deploymentJson | ConvertFrom-Json

$acrLoginServer = $deploymentOutput.acrLoginServer.value
$containerAppName = $deploymentOutput.containerAppName.value
$containerAppUrl = $deploymentOutput.containerAppUrl.value
$principalId = $deploymentOutput.principalId.value

Write-Host "   ACR: $acrLoginServer"
Write-Host "   Container App: $containerAppName"
Write-Host "   Principal ID: $principalId"

# Build and push container image
Write-Host "`nBuilding and pushing container image..." -ForegroundColor Yellow
$acrName = $acrLoginServer.Split('.')[0]

az acr build --registry $acrName --resource-group $ResourceGroup --image "${containerAppName}:$ImageTag" --file "$ScriptDir/Dockerfile.agent-api" $ScriptDir

# Update container app to use new image (triggers deployment)
Write-Host "`nUpdating container app..." -ForegroundColor Yellow
az containerapp update --name $containerAppName --resource-group $ResourceGroup --image "$acrLoginServer/${containerAppName}:$ImageTag" --output none

# Assign RBAC to AI Project (cross-resource-group)
Write-Host "`nAssigning RBAC to AI Project..." -ForegroundColor Yellow
$subscriptionId = $account.id
$aiResourceId = "/subscriptions/$subscriptionId/resourceGroups/$AiResourceGroup/providers/Microsoft.CognitiveServices/accounts/$AiResourceName"

az role assignment create --assignee $principalId --role "Cognitive Services User" --scope $aiResourceId --output none 2>$null

Write-Host "`nDeployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "   API URL: $containerAppUrl" -ForegroundColor Cyan
Write-Host "   API Docs: $containerAppUrl/docs"
Write-Host "   Health: $containerAppUrl/health"
Write-Host ""
Write-Host "Test with:" -ForegroundColor Yellow
Write-Host "Invoke-RestMethod -Uri '$containerAppUrl/chat' -Method Post -ContentType 'application/json' -Body '{""message"":""Hello!""}'"
