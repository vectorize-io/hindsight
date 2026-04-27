package io.vectorize.hindsight.android

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import kotlinx.coroutines.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val baseUrl = "http://127.0.0.1:8741"
    private var bankId: String? = null

    private lateinit var statusText: TextView
    private lateinit var apiKeyInput: EditText
    private lateinit var retainInput: EditText
    private lateinit var queryInput: EditText
    private lateinit var resultsText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        apiKeyInput = findViewById(R.id.apiKeyInput)
        retainInput = findViewById(R.id.retainInput)
        queryInput = findViewById(R.id.queryInput)
        resultsText = findViewById(R.id.resultsText)

        findViewById<Button>(R.id.startButton).setOnClickListener { startServer() }
        findViewById<Button>(R.id.retainButton).setOnClickListener { retainMemory() }
        findViewById<Button>(R.id.recallButton).setOnClickListener { recallMemory() }

        statusText.text = "Server: stopped"
    }

    private fun startServer() {
        val apiKey = apiKeyInput.text.toString().trim()
        if (apiKey.isEmpty()) {
            Toast.makeText(this, "Enter an OpenAI API key", Toast.LENGTH_SHORT).show()
            return
        }

        statusText.text = "Server: starting..."

        scope.launch(Dispatchers.IO) {
            try {
                val py = Python.getInstance()
                val server = py.getModule("hindsight_android.server")
                val dbPath = "${filesDir.absolutePath}/hindsight.db"

                server.callAttr("start_server", dbPath, apiKey, "gpt-4o-mini", 8741)

                // Wait for server to be ready
                var ready = false
                for (i in 1..30) {
                    delay(500)
                    try {
                        val req = Request.Builder().url("$baseUrl/health").build()
                        val resp = client.newCall(req).execute()
                        if (resp.isSuccessful) {
                            ready = true
                            resp.close()
                            break
                        }
                        resp.close()
                    } catch (_: Exception) {}
                }

                if (ready) {
                    // Create a default bank
                    val body = JSONObject().apply {
                        put("name", "android-demo")
                        put("mission", "On-device memory demo")
                    }
                    val req = Request.Builder()
                        .url("$baseUrl/v1/default/banks")
                        .post(body.toString().toRequestBody("application/json".toMediaType()))
                        .build()
                    val resp = client.newCall(req).execute()
                    val respBody = resp.body?.string() ?: "{}"
                    resp.close()
                    bankId = JSONObject(respBody).optString("bank_id")

                    withContext(Dispatchers.Main) {
                        statusText.text = "Server: running ✓\nBank: $bankId"
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Server: failed to start"
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun retainMemory() {
        val content = retainInput.text.toString().trim()
        if (content.isEmpty() || bankId == null) {
            Toast.makeText(this, "Enter content and start server first", Toast.LENGTH_SHORT).show()
            return
        }

        resultsText.text = "Retaining..."

        scope.launch(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("items", JSONArray().apply {
                        put(JSONObject().apply { put("content", content) })
                    })
                }

                val req = Request.Builder()
                    .url("$baseUrl/v1/default/banks/$bankId/memories/retain")
                    .post(body.toString().toRequestBody("application/json".toMediaType()))
                    .build()

                val resp = client.newCall(req).execute()
                val respBody = resp.body?.string() ?: "{}"
                resp.close()
                val result = JSONObject(respBody)

                withContext(Dispatchers.Main) {
                    val facts = result.optInt("facts_extracted", 0)
                    resultsText.text = "Retained: $facts facts extracted"
                    retainInput.text.clear()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    resultsText.text = "Retain error: ${e.message}"
                }
            }
        }
    }

    private fun recallMemory() {
        val query = queryInput.text.toString().trim()
        if (query.isEmpty() || bankId == null) {
            Toast.makeText(this, "Enter a query and start server first", Toast.LENGTH_SHORT).show()
            return
        }

        resultsText.text = "Searching..."

        scope.launch(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("query", query)
                    put("max_results", 10)
                }

                val req = Request.Builder()
                    .url("$baseUrl/v1/default/banks/$bankId/memories/recall")
                    .post(body.toString().toRequestBody("application/json".toMediaType()))
                    .build()

                val resp = client.newCall(req).execute()
                val respBody = resp.body?.string() ?: "{}"
                resp.close()
                val result = JSONObject(respBody)

                val results = result.optJSONArray("results") ?: JSONArray()
                val sb = StringBuilder()
                sb.appendLine("Found ${results.length()} memories:\n")

                for (i in 0 until results.length()) {
                    val r = results.getJSONObject(i)
                    val score = r.optDouble("score", 0.0)
                    val text = r.optString("text", "")
                    val type = r.optString("type", "")
                    sb.appendLine("${i + 1}. [$type] (${String.format("%.2f", score)}) $text")
                    sb.appendLine()
                }

                withContext(Dispatchers.Main) {
                    resultsText.text = sb.toString()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    resultsText.text = "Recall error: ${e.message}"
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        try {
            val py = Python.getInstance()
            val server = py.getModule("hindsight_android.server")
            server.callAttr("stop_server")
        } catch (_: Exception) {}
    }
}
