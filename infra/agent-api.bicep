// Hindsight Agent API - Azure Container Apps Infrastructure
// Bicep template for deploying the agent API alongside existing hindsight-api

@description('Location for all resources')
param location string = 'centralus'

@description('Resource group containing the AI Project')
param aiProjectResourceGroup string = 'jacob-1216-resource'

@description('Name of the AI resource for RBAC')
param aiResourceName string = 'jacob-1216-resource'

@description('Container image tag')
param imageTag string = 'latest'

// Naming
var containerAppName = 'hindsight-agent-api'
var containerAppEnvName = 'hindsight-env'
var acrName = 'hindsightacr${uniqueString(resourceGroup().id)}'
var logAnalyticsName = 'hindsight-logs'

// Configuration
var hindsightApiUrl = 'https://hindsight-api.politebay-1635b4f9.centralus.azurecontainerapps.io'
var projectEndpoint = 'https://jacob-1216-resource.services.ai.azure.com/api/projects/jacob-1216'
var defaultBankId = 'hindsight_agent_bank'

// Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// Container Apps Environment
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Container App - Hindsight Agent API
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        transport: 'http'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          name: containerAppName
          image: '${acr.properties.loginServer}/${containerAppName}:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'HINDSIGHT_PROJECT_ENDPOINT'
              value: projectEndpoint
            }
            {
              name: 'HINDSIGHT_MCP_BASE_URL'
              value: hindsightApiUrl
            }
            {
              name: 'HINDSIGHT_DEFAULT_BANK_ID'
              value: defaultBankId
            }
            {
              name: 'PORT'
              value: '8080'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8080
              }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8080
              }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

// Role Assignment - Cognitive Services User on AI Project
// This requires the AI resource to be in the same subscription
resource cognitiveServicesRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerApp.id, 'CognitiveServicesUser', aiResourceName)
  scope: resourceGroup()
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908') // Cognitive Services User
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Outputs
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output containerAppName string = containerApp.name
output acrLoginServer string = acr.properties.loginServer
output principalId string = containerApp.identity.principalId
