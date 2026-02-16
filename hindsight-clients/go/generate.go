package hindsight

//go:generate go run ./internal/cmd/preprocess ../../hindsight-docs/static/openapi.json internal/ogenapi/openapi.json
//go:generate go run github.com/ogen-go/ogen/cmd/ogen --target internal/ogenapi -package ogenapi --clean --config ogen.yml internal/ogenapi/openapi.json
//go:generate bash ./patch-ogen.sh
