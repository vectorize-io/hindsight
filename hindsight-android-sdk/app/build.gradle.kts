plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "io.vectorize.hindsight.android"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.vectorize.hindsight.android"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-poc"

        ndk {
            abiFilters += "arm64-v8a"
        }

        python {
            version = "3.11"

            pip {
                // Pure Python packages from PyPI
                install("fastapi==0.120.3")
                install("uvicorn==0.38.0")
                install("pydantic==2.11.7")
                install("openai==1.87.0")
                install("httpx==0.28.1")
                install("starlette")
                install("h11")
                install("sniffio")
                install("anyio")
                install("typing-extensions")
                install("annotated-types")
                install("typing_inspection")
                install("annotated-doc")
                install("idna")
                install("certifi")
                install("aiosqlite")
                install("tqdm")
            }

            extractPackages("pydantic_core")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
