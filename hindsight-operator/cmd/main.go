// Command manager runs the Hindsight operator.
package main

import (
	"flag"
	"net/http"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	hindsightv1alpha1 "github.com/vectorize-io/hindsight/hindsight-operator/api/v1alpha1"
	"github.com/vectorize-io/hindsight/hindsight-operator/internal/controller"
)

func main() {
	var (
		metricsAddr    string
		probeAddr      string
		leaderElect    bool
		resyncPeriod   time.Duration
		apiTimeout     time.Duration
		watchNamespace string
	)
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "Address the metrics endpoint binds to. Use 0 to disable it.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "Address the health probe endpoint binds to.")
	flag.BoolVar(&leaderElect, "leader-elect", true, "Use leader election so that only one replica reconciles.")
	flag.DurationVar(&resyncPeriod, "resync-period", 10*time.Minute, "How often each template is imported again, which restores settings changed outside Kubernetes.")
	flag.DurationVar(&apiTimeout, "api-timeout", 60*time.Second, "Timeout for each Hindsight API request.")
	flag.StringVar(&watchNamespace, "watch-namespace", "", "Only reconcile HindsightBanks in this namespace. Empty watches all namespaces.")
	opts := zap.Options{}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))
	setupLog := ctrl.Log.WithName("setup")

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(hindsightv1alpha1.AddToScheme(scheme))

	options := ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsserver.Options{BindAddress: metricsAddr},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         leaderElect,
		LeaderElectionID:       "hindsight-operator.hindsight.vectorize.io",
	}
	if watchNamespace != "" {
		options.Cache.DefaultNamespaces = map[string]cache.Config{watchNamespace: {}}
	}
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), options)
	if err != nil {
		setupLog.Error(err, "unable to create manager")
		os.Exit(1)
	}

	if err := (&controller.HindsightBankReconciler{
		Client:       mgr.GetClient(),
		APIReader:    mgr.GetAPIReader(),
		HTTPClient:   &http.Client{Timeout: apiTimeout},
		ResyncPeriod: resyncPeriod,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "HindsightBank")
		os.Exit(1)
	}

	utilruntime.Must(mgr.AddHealthzCheck("healthz", healthz.Ping))
	utilruntime.Must(mgr.AddReadyzCheck("readyz", healthz.Ping))

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "manager exited with an error")
		os.Exit(1)
	}
}
