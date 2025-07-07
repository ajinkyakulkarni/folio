---
title: "Kubernetes Security: A Comprehensive Guide to Hardening Your Clusters"
description: "Master Kubernetes security from pod security policies to network segmentation. Learn how to protect your clusters against common threats and implement defense in depth."
author: sarah-johnson
publishDate: 2024-03-17
heroImage: https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=800&h=400&fit=crop
category: "DevOps"
tags: ["kubernetes", "security", "devops", "containers", "cloud"]
featured: true
draft: false
readingTime: 18
---

## Introduction

Kubernetes security is complex but critical. This guide provides practical steps to secure your clusters, from basic hardening to advanced threat protection.

## Security Layers in Kubernetes

1. **Infrastructure Security**
2. **Cluster Security**
3. **Container Security**
4. **Application Security**

## RBAC: Your First Line of Defense

### Setting Up Role-Based Access Control

```yaml
# developer-role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: developer
rules:
- apiGroups: ["apps", ""]
  resources: ["deployments", "pods", "services"]
  verbs: ["get", "list", "watch", "create", "update"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-binding
  namespace: development
subjects:
- kind: User
  name: jane@example.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: developer
  apiGroup: rbac.authorization.k8s.io
```

### Service Account Security

```yaml
# Disable automounting of service account tokens
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
automountServiceAccountToken: false
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
spec:
  template:
    spec:
      serviceAccountName: app-service-account
      automountServiceAccountToken: false
      containers:
      - name: app
        image: myapp:latest
```

## Pod Security Standards

### Implementing Pod Security Policies

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### Security Contexts

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-deployment
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: app
        image: myapp:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL
            add:
              - NET_BIND_SERVICE
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}
```

## Network Security

### Network Policies

```yaml
# Default deny all traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
# Allow specific traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: web-netpol
spec:
  podSelector:
    matchLabels:
      app: web
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
```

### Service Mesh Security with Istio

```yaml
# Enable mTLS for service-to-service communication
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: STRICT
---
# Authorization policy
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: frontend-to-backend
  namespace: production
spec:
  selector:
    matchLabels:
      app: backend
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/production/sa/frontend"]
    to:
    - operation:
        methods: ["GET", "POST"]
```

## Secrets Management

### Using Sealed Secrets

```bash
# Install sealed-secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

# Create a secret
echo -n mypassword | kubectl create secret generic mysecret --dry-run=client --from-file=password=/dev/stdin -o yaml | kubeseal -o yaml > mysealedsecret.yaml
```

### External Secrets with HashiCorp Vault

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "demo"
          serviceAccountRef:
            name: "vault-auth"
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: vault-secret
spec:
  refreshInterval: 15s
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: example-secret
  data:
  - secretKey: password
    remoteRef:
      key: secret/data/database
      property: password
```

## Container Image Security

### Scanning Images in CI/CD

```yaml
# GitLab CI example
image_scan:
  stage: test
  image: aquasec/trivy:latest
  script:
    - trivy image --severity HIGH,CRITICAL --exit-code 1 $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  allow_failure: false
```

### Admission Controllers with OPA

```rego
# policy.rego
package kubernetes.admission

deny[msg] {
  input.request.kind.kind == "Pod"
  input.request.object.spec.containers[_].image
  not starts_with(input.request.object.spec.containers[_].image, "registry.company.com/")
  msg := "Images must be from approved registry"
}

deny[msg] {
  input.request.kind.kind == "Pod"
  input.request.object.spec.containers[_].securityContext.privileged == true
  msg := "Privileged containers are not allowed"
}
```

## Monitoring and Auditing

### Enable Audit Logging

```yaml
# audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  # Don't log read-only requests
  - level: None
    verbs: ["get", "list", "watch"]
  
  # Log pod changes at Metadata level
  - level: Metadata
    omitStages:
      - RequestReceived
    resources:
      - group: ""
        resources: ["pods", "services"]
    namespaces: ["production", "staging"]
  
  # Log everything else at RequestResponse level
  - level: RequestResponse
    omitStages:
      - RequestReceived
```

### Security Monitoring with Falco

```yaml
# falco-rules.yaml
- rule: Terminal shell in container
  desc: A shell was used as the entrypoint/exec
  condition: >
    spawned_process and container
    and shell_procs and proc.name in (shell_binaries)
    and not user_expected_terminal_shell_in_container
  output: >
    Terminal shell in container
    (user=%user.name container_id=%container.id container_name=%container.name
    shell=%proc.name parent=%proc.pname cmdline=%proc.cmdline)
  priority: WARNING
```

## Supply Chain Security

### Signing Images with Cosign

```bash
# Generate keys
cosign generate-key-pair

# Sign image
cosign sign --key cosign.key registry.company.com/myapp:v1.0

# Verify signature
cosign verify --key cosign.pub registry.company.com/myapp:v1.0
```

### Policy Enforcement with Gatekeeper

```yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: k8srequiredlabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredLabels
      validation:
        openAPIV3Schema:
          type: object
          properties:
            labels:
              type: array
              items:
                type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8srequiredlabels
        violation[{"msg": msg}] {
          required := input.parameters.labels
          provided := input.review.object.metadata.labels
          missing := required[_]
          not provided[missing]
          msg := sprintf("Label '%v' is required", [missing])
        }
```

## Incident Response

### Security Runbook Template

```bash
#!/bin/bash
# incident-response.sh

# 1. Isolate the affected pod
kubectl cordon $NODE_NAME
kubectl label pod $POD_NAME quarantine=true

# 2. Capture forensic data
kubectl logs $POD_NAME > incident-$TIMESTAMP.log
kubectl describe pod $POD_NAME > incident-$TIMESTAMP-describe.log
kubectl exec $POD_NAME -- ps aux > incident-$TIMESTAMP-processes.log

# 3. Check for persistence
kubectl get cronjobs,jobs,deployments,daemonsets,statefulsets -A | grep -i $SUSPICIOUS_PATTERN

# 4. Review recent changes
kubectl get events --sort-by='.lastTimestamp' -A
```

## Security Checklist

- [ ] RBAC properly configured
- [ ] Pod Security Standards enforced
- [ ] Network policies implemented
- [ ] Secrets encrypted at rest
- [ ] Image scanning in CI/CD
- [ ] Admission controllers deployed
- [ ] Audit logging enabled
- [ ] Runtime security monitoring
- [ ] Regular security updates
- [ ] Incident response plan tested

## Conclusion

Kubernetes security requires a defense-in-depth approach. Start with the basics—RBAC and pod security—then layer on additional controls. Regular audits and updates are essential to maintain security posture. Remember, security is not a destination but a continuous journey.