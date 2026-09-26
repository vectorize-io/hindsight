ENVTEST_K8S_VERSION ?= 1.35.x
LOCALBIN ?= $(CURDIR)/bin

.PHONY: generate
generate: ## Regenerate deepcopy code, the CRD, and RBAC from the Go markers.
	go tool controller-gen object paths=./api/...
	go tool controller-gen crd rbac:roleName=hindsight-operator paths=./... \
		output:crd:artifacts:config=config/crd/bases output:rbac:artifacts:config=config/rbac

.PHONY: check-generated
check-generated: generate ## Fail when committed generated files are stale.
	git diff --exit-code -- api config

.PHONY: test
test: ## Run unit tests and controller tests against a local API server.
	KUBEBUILDER_ASSETS="$$(go tool setup-envtest use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test ./... -count=1

.PHONY: build
build:
	go build -o $(LOCALBIN)/manager ./cmd

.PHONY: lint
lint:
	@out="$$(gofmt -l .)"; if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	go vet ./...
