package com.vrbox.ar

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.net.Uri
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.opengl.Matrix
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.view.View
import android.view.WindowManager
import android.widget.EditText
import android.widget.Toast
import com.google.ar.core.*
import com.google.ar.core.exceptions.*
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.*

class Btn(val x: Float, val y: Float, val w: Float, val h: Float, val l: String, val fs: Float = 26f, val fn: () -> Unit)

class Win(val id: String, val title: String, val lw: Int, val lh: Int, var w: Float, var pos: FloatArray) {
    var hover = -1
    var btns: List<Btn> = emptyList()
    var tex = 0
    val bmp: Bitmap = Bitmap.createBitmap(lw, lh, Bitmap.Config.ARGB_8888)
    val cv = Canvas(bmp)
    var sig = ""
    val hm get() = w * lh / lw
}

class Hand {
    var on = false; var x = 0f; var y = 0f; var sz = .1f
    var pinch = false; var down = false; var cnt = 0; var miss = 0; var seq = -1L
    var nx = 0f; var ny = 0f
    var grab: Win? = null; var t0 = 1f; var sz0 = .1f; val off = FloatArray(3)
}

class MainActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var gv: GLSurfaceView
    private var session: Session? = null
    private var wantInstall = true
    private var vw = 1; private var vh = 1; private var geomDirty = true; private var spawned = false
    private val view = FloatArray(16); private val proj = FloatArray(16); private val vpm = FloatArray(16)
    private val inv = FloatArray(16); private val ev = FloatArray(16); private val tm = FloatArray(16)
    private val evp = FloatArray(16); private val idm = FloatArray(16)
    private var cam = FloatArray(3); private var fwd = floatArrayOf(0f, 0f, -1f)
    private val wins = ArrayList<Win>()
    private var ipd = .063f; private var stereo = true; private var fov = 70f
    private var foc = "note"; private var note = ""; private var ai = ""; private var shift = false; private var aiBusy = false
    private val chat = CopyOnWriteArrayList<String>()
    private var apiKey = ""
    // GL
    private var bgTex = 0; private var bgP = 0; private var pnP = 0
    private val bgPos: FloatBuffer = fb(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
    private val bgUv: FloatBuffer = fb(FloatArray(8))
    private val pv: FloatBuffer = fb(FloatArray(20))
    private val curTex = IntArray(2)
    // hands
    private var lm: HandLandmarker? = null
    private val ex = Executors.newSingleThreadExecutor()
    @Volatile private var busy = false
    @Volatile private var raw: List<FloatArray> = emptyList()
    @Volatile private var rawSeq = 0L
    private val hands = listOf(Hand(), Hand())
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    private fun fb(a: FloatArray): FloatBuffer = ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(a); position(0) }

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        gv = GLSurfaceView(this)
        gv.preserveEGLContextOnPause = true
        gv.setEGLContextClientVersion(2)
        gv.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        gv.setRenderer(this)
        gv.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        setContentView(gv)
        apiKey = getPreferences(MODE_PRIVATE).getString("gk", "") ?: ""
        if (apiKey.isEmpty()) askKey()
        Matrix.setIdentityM(idm, 0)
    }

    private fun askKey() = runOnUiThread {
        val e = EditText(this); e.hint = "Gemini API anahtarı (AIza...)"
        AlertDialog.Builder(this).setTitle("Gemini API anahtarı").setView(e)
            .setPositiveButton("Kaydet") { _, _ -> apiKey = e.text.toString().trim(); getPreferences(MODE_PRIVATE).edit().putString("gk", apiKey).apply() }
            .setNegativeButton("Atla", null).show()
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1); return
        }
        try {
            if (session == null) {
                if (ArCoreApk.getInstance().requestInstall(this, wantInstall) == ArCoreApk.InstallStatus.INSTALL_REQUESTED) { wantInstall = false; return }
                val s = Session(this)
                val cfg = Config(s)
                cfg.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                cfg.focusMode = Config.FocusMode.AUTO
                cfg.planeFindingMode = Config.PlaneFindingMode.DISABLED
                cfg.lightEstimationMode = Config.LightEstimationMode.DISABLED
                s.configure(cfg)
                val l = s.getSupportedCameraConfigs(CameraConfigFilter(s).setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_60)))
                if (l.isNotEmpty()) s.cameraConfig = l[0]
                session = s
                Thread {
                    try {
                        lm = HandLandmarker.createFromOptions(this, HandLandmarker.HandLandmarkerOptions.builder()
                            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
                            .setNumHands(2).setRunningMode(RunningMode.IMAGE).build())
                    } catch (e: Exception) { toast("El takibi yüklenemedi: ${e.message}") }
                }.start()
            }
            session!!.resume()
            gv.onResume()
        } catch (e: UnavailableUserDeclinedInstallationException) { toast("ARCore gerekli"); finish()
        } catch (e: Exception) { toast("ARCore hatası: ${e.message}"); finish() }
    }

    override fun onRequestPermissionsResult(c: Int, p: Array<out String>, r: IntArray) {
        if (r.isNotEmpty() && r[0] == PackageManager.PERMISSION_GRANTED) recreate() else { toast("Kamera izni şart"); finish() }
    }

    override fun onPause() { super.onPause(); gv.onPause(); session?.pause() }
    private fun toast(s: String) = runOnUiThread { Toast.makeText(this, s, Toast.LENGTH_LONG).show() }
    private fun vib() { try { (getSystemService(VIBRATOR_SERVICE) as Vibrator).vibrate(VibrationEffect.createOneShot(15, 120)) } catch (e: Exception) {} }

    // ---------------- GL ----------------
    private fun sh(t: Int, s: String): Int { val i = GLES20.glCreateShader(t); GLES20.glShaderSource(i, s); GLES20.glCompileShader(i); return i }
    private fun prog(v: String, f: String): Int { val p = GLES20.glCreateProgram(); GLES20.glAttachShader(p, sh(GLES20.GL_VERTEX_SHADER, v)); GLES20.glAttachShader(p, sh(GLES20.GL_FRAGMENT_SHADER, f)); GLES20.glLinkProgram(p); return p }

    override fun onSurfaceCreated(g: GL10?, c: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        val t = IntArray(1); GLES20.glGenTextures(1, t, 0); bgTex = t[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, bgTex)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        bgP = prog("attribute vec2 aPos;attribute vec2 aUv;varying vec2 vUv;void main(){gl_Position=vec4(aPos,0.,1.);vUv=aUv;}",
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float;varying vec2 vUv;uniform samplerExternalOES uTex;void main(){gl_FragColor=texture2D(uTex,vUv);}")
        pnP = prog("attribute vec3 aPos;attribute vec2 aUv;uniform mat4 uM;varying vec2 vUv;void main(){gl_Position=uM*vec4(aPos,1.);vUv=aUv;}",
            "precision mediump float;varying vec2 vUv;uniform sampler2D uTex;void main(){gl_FragColor=texture2D(uTex,vUv);}")
        for (i in 0..1) {
            val b = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888); val cc = Canvas(b)
            paint.style = Paint.Style.FILL; paint.color = if (i == 0) 0xCC7DFFB0.toInt() else 0xFFFF4F6A.toInt(); cc.drawCircle(32f, 32f, if (i == 0) 26f else 18f, paint)
            paint.style = Paint.Style.STROKE; paint.strokeWidth = 5f; paint.color = Color.WHITE; cc.drawCircle(32f, 32f, if (i == 0) 26f else 18f, paint)
            curTex[i] = mkTex(b)
        }
        wins.forEach { it.tex = 0; it.sig = "" }
    }

    private fun mkTex(b: Bitmap): Int {
        val t = IntArray(1); GLES20.glGenTextures(1, t, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, b, 0); return t[0]
    }

    override fun onSurfaceChanged(g: GL10?, w: Int, h: Int) { vw = w; vh = h; geomDirty = true }

    override fun onDrawFrame(g: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val s = session ?: return
        if (geomDirty) { s.setDisplayGeometry(windowManager.defaultDisplay.rotation, if (stereo) vw / 2 else vw, vh); geomDirty = false }
        s.setCameraTextureName(bgTex)
        val frame = try { s.update() } catch (e: Exception) { return }
        if (frame.hasDisplayGeometryChanged()) {
            bgPos.position(0); bgUv.position(0)
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, bgPos, Coordinates2d.TEXTURE_NORMALIZED, bgUv)
        }
        val camera = frame.camera
        val trk = camera.trackingState == TrackingState.TRACKING
        val ew = if (stereo) vw / 2 else vw
        if (trk) {
            camera.getViewMatrix(view, 0)
            val p = camera.displayOrientedPose
            cam = p.translation; val z = p.zAxis; fwd = floatArrayOf(-z[0], -z[1], -z[2])
            Matrix.perspectiveM(proj, 0, fov, ew.toFloat() / vh, .05f, 50f)
            Matrix.multiplyMM(vpm, 0, proj, 0, view, 0); Matrix.invertM(inv, 0, vpm, 0)
            if (!spawned) { spawn(); spawned = true }
            updateHands(frame)
        }
        eye(0, ew, if (stereo) ipd / 2 else 0f, trk)
        if (stereo) eye(ew, ew, -ipd / 2, trk)
    }

    private fun eye(x: Int, w: Int, sx: Float, trk: Boolean) {
        GLES20.glViewport(x, 0, w, vh)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST); GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glUseProgram(bgP)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0); GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, bgTex)
        val a = GLES20.glGetAttribLocation(bgP, "aPos"); val u = GLES20.glGetAttribLocation(bgP, "aUv")
        bgPos.position(0); bgUv.position(0)
        GLES20.glVertexAttribPointer(a, 2, GLES20.GL_FLOAT, false, 0, bgPos); GLES20.glEnableVertexAttribArray(a)
        GLES20.glVertexAttribPointer(u, 2, GLES20.GL_FLOAT, false, 0, bgUv); GLES20.glEnableVertexAttribArray(u)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        if (!trk) return
        Matrix.setIdentityM(tm, 0); Matrix.translateM(tm, 0, sx, 0f, 0f)
        Matrix.multiplyMM(ev, 0, tm, 0, view, 0); Matrix.multiplyMM(evp, 0, proj, 0, ev, 0)
        GLES20.glEnable(GLES20.GL_BLEND); GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glUseProgram(pnP)
        val pa = GLES20.glGetAttribLocation(pnP, "aPos"); val pu = GLES20.glGetAttribLocation(pnP, "aUv")
        val um = GLES20.glGetUniformLocation(pnP, "uM")
        for (wn in wins.toList().sortedByDescending { dist(it) }) {
            val sg = "${wn.hover}|${if (foc == wn.id) System.currentTimeMillis() / 500 % 2 else 2}|$note|$ai|${chat.size}|$aiBusy|$shift|$ipd|$stereo|$fov"
            if (wn.tex == 0) { wn.tex = mkTex(wn.bmp); wn.sig = "" }
            if (wn.sig != sg) { render(wn); wn.sig = sg; GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, wn.tex); GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, wn.bmp, 0) }
            val n = nrm(wn); val r = floatArrayOf(n[2], 0f, -n[0]); val hw = wn.w / 2; val hh = wn.hm / 2
            val c = wn.pos
            val v = arrayOf(floatArrayOf(-1f, -1f, 0f, 1f), floatArrayOf(1f, -1f, 1f, 1f), floatArrayOf(-1f, 1f, 0f, 0f), floatArrayOf(1f, 1f, 1f, 0f))
            pv.position(0)
            for (q in v) { pv.put(c[0] + r[0] * hw * q[0]); pv.put(c[1] + hh * q[1]); pv.put(c[2] + r[2] * hw * q[0]); pv.put(q[2]); pv.put(q[3]) }
            quad(pa, pu, um, evp, wn.tex)
        }
        for (h in hands) if (h.on) {
            val sy = .05f; val sxx = sy * vh / w
            pv.position(0)
            val vs = arrayOf(floatArrayOf(-1f, -1f, 0f, 1f), floatArrayOf(1f, -1f, 1f, 1f), floatArrayOf(-1f, 1f, 0f, 0f), floatArrayOf(1f, 1f, 1f, 0f))
            for (q in vs) { pv.put(h.nx + sxx * q[0]); pv.put(h.ny + sy * q[1]); pv.put(0f); pv.put(q[2]); pv.put(q[3]) }
            quad(pa, pu, um, idm, curTex[if (h.pinch) 1 else 0])
        }
    }

    private fun quad(pa: Int, pu: Int, um: Int, m: FloatArray, tex: Int) {
        GLES20.glUniformMatrix4fv(um, 1, false, m, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0); GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        pv.position(0); GLES20.glVertexAttribPointer(pa, 3, GLES20.GL_FLOAT, false, 20, pv); GLES20.glEnableVertexAttribArray(pa)
        pv.position(3); GLES20.glVertexAttribPointer(pu, 2, GLES20.GL_FLOAT, false, 20, pv); GLES20.glEnableVertexAttribArray(pu)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }

    private fun dist(w: Win) = hypot(hypot((w.pos[0] - cam[0]).toDouble(), (w.pos[1] - cam[1]).toDouble()), (w.pos[2] - cam[2]).toDouble())
    private fun nrm(w: Win): FloatArray { var x = cam[0] - w.pos[0]; var z = cam[2] - w.pos[2]; val l = hypot(x, z).coerceAtLeast(1e-4f); x /= l; z /= l; return floatArrayOf(x, 0f, z) }

    // ---------------- Pencereler ----------------
    private fun spawn() {
        wins.add(Win("dock", "", 640, 100, .75f, at(0f, -.38f, 1.3f)))
        open("note")
    }
    private fun at(dx: Float, dy: Float, d: Float): FloatArray {
        val l = hypot(fwd[0], fwd[2]).coerceAtLeast(1e-4f); val fx = fwd[0] / l; val fz = fwd[2] / l
        return floatArrayOf(cam[0] + fx * d - fz * dx, cam[1] + dy, cam[2] + fz * d + fx * dx)
    }
    private fun open(id: String) {
        val old = wins.find { it.id == id }
        val (t, lw, lh, w) = when (id) {
            "note" -> listOf("📝 Not", 480, 300, .8f); "kb" -> listOf("⌨ Klavye", 540, 262, .8f)
            "ai" -> listOf("🤖 Gemini", 480, 350, .85f); else -> listOf("⚙ Ayarlar", 420, 280, .7f) }
        if (id == "note" || id == "ai") foc = id
        val p = at(0f, if (id == "kb") -.3f else .05f, 1.4f)
        if (old != null) { old.pos = p; return }
        wins.add(Win(id, t as String, lw as Int, lh as Int, w as Float, p))
        if ((id == "note" || id == "ai") && wins.none { it.id == "kb" }) open("kb")
    }
    private fun extUri(u: String) = runOnUiThread { try { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(u))) } catch (e: Exception) { toast("Açılamadı") } }

    private fun txt(c: Canvas, s: String, x: Float, y: Float, sz: Float, al: Paint.Align = Paint.Align.LEFT, col: Int = Color.WHITE) {
        paint.style = Paint.Style.FILL; paint.color = col; paint.textSize = sz; paint.textAlign = al; c.drawText(s, x, y + sz * .35f, paint)
    }
    private fun wrap(s: String, n: Int): List<String> { val o = ArrayList<String>(); s.split("\n").forEach { var l = it; do { o.add(l.take(n)); l = l.drop(n) } while (l.isNotEmpty()) }; return o }

    private fun render(w: Win) {
        val c = w.cv; c.drawColor(0, PorterDuff.Mode.CLEAR)
        val b = ArrayList<Btn>(); val W = w.lw.toFloat(); val H = w.lh.toFloat()
        paint.style = Paint.Style.FILL; paint.color = 0xCC1E2438.toInt(); c.drawRoundRect(0f, 0f, W, H, 32f, 32f, paint)
        paint.style = Paint.Style.STROKE; paint.strokeWidth = 3f; paint.color = 0x66FFFFFF; c.drawRoundRect(1f, 1f, W - 1, H - 1, 32f, 32f, paint)
        val blink = System.currentTimeMillis() / 500 % 2 == 0L
        if (w.id != "dock") { txt(c, w.title, 22f, 28f, 24f); b.add(Btn(W - 56, 8f, 44f, 40f, "✕", 22f) { wins.remove(w) }) }
        when (w.id) {
            "dock" -> listOf("🤖" to { open("ai") }, "🌐" to { extUri("https://www.google.com") }, "🛍" to { extUri("market://search?q=vr%20games&c=apps") },
                "📝" to { open("note") }, "⌨" to { open("kb") }, "⚙" to { open("set") }).forEachIndexed { i, p -> b.add(Btn(18f + i * 102f, 14f, 88f, 72f, p.first, 40f, p.second)) }
            "note" -> { wrap(note + if (foc == "note" && blink) "|" else "", 34).takeLast(8).forEachIndexed { i, l -> txt(c, l, 22f, 78f + i * 26f, 22f) }
                b.add(Btn(W - 150, 8f, 90f, 40f, "Temizle", 16f) { note = "" }) }
            "ai" -> { val ln = ArrayList<String>(); chat.takeLast(6).forEach { ln.addAll(wrap(it, 36)) }; if (aiBusy) ln.add("… düşünüyor")
                ln.takeLast(9).forEachIndexed { i, l -> txt(c, l, 20f, 76f + i * 25f, 21f) }
                txt(c, "💬 " + ai + if (foc == "ai" && blink) "|" else "", 20f, 318f - 26f, 21f, col = if (foc == "ai") 0xFF9FD0FF.toInt() else Color.WHITE)
                b.add(Btn(20f, 300f, 130f, 40f, "Gönder", 20f) { askAI() }); b.add(Btn(W - 150, 8f, 90f, 40f, "Temizle", 16f) { chat.clear() }) }
            "kb" -> { listOf("1234567890", "qwertyuıopğü", "asdfghjklşi", "zxcvbnmöç.-").forEachIndexed { n, r -> r.forEachIndexed { j, ch ->
                    b.add(Btn(14f + j * 42f + n * 10f, 54f + n * 46f, 40f, 42f, if (shift) ch.uppercase() else ch.toString(), 22f) { key(ch.toString()) }) } }
                b.add(Btn(14f, 222f, 70f, 36f, "⇧", 22f) { shift = !shift }); b.add(Btn(94f, 222f, 250f, 36f, "boşluk", 20f) { key(" ") })
                b.add(Btn(354f, 222f, 80f, 36f, "⌫", 22f) { key("⌫") }); b.add(Btn(444f, 222f, 80f, 36f, "⏎", 22f) { key("⏎") }) }
            else -> { txt(c, "IPD: ${(ipd * 1000).roundToInt()} mm", 22f, 84f, 22f); b.add(Btn(250f, 62f, 60f, 44f, "−", 26f) { ipd = (ipd - .002f).coerceIn(.04f, .08f) }); b.add(Btn(320f, 62f, 60f, 44f, "+", 26f) { ipd = (ipd + .002f).coerceIn(.04f, .08f) })
                txt(c, "Görüş: ${fov.roundToInt()}°", 22f, 138f, 22f); b.add(Btn(250f, 116f, 60f, 44f, "−", 26f) { fov = (fov - 5).coerceIn(40f, 110f) }); b.add(Btn(320f, 116f, 60f, 44f, "+", 26f) { fov = (fov + 5).coerceIn(40f, 110f) })
                b.add(Btn(20f, 176f, 190f, 44f, if (stereo) "Çift ekran: Açık" else "Çift ekran: Kapalı", 17f) { stereo = !stereo; geomDirty = true })
                b.add(Btn(220f, 176f, 180f, 44f, "Gemini anahtarı", 17f) { askKey() }); b.add(Btn(20f, 228f, 190f, 40f, "Ortala", 18f) { val d = wins.toList(); d.forEach { if (it.id != "set") it.pos = at(0f, if (it.id == "dock") -.38f else if (it.id == "kb") -.3f else .05f, 1.4f) } })
                b.add(Btn(220f, 228f, 180f, 40f, "Hepsini kapat", 17f) { wins.removeAll { it.id != "dock" && it.id != "set" } }) }
        }
        b.forEachIndexed { i, k ->
            paint.style = Paint.Style.FILL; paint.color = if (w.hover == i) 0x99FFFFFF.toInt() else 0x33FFFFFF; c.drawRoundRect(k.x, k.y, k.x + k.w, k.y + k.h, k.h * .3f, k.h * .3f, paint)
            txt(c, k.l, k.x + k.w / 2, k.y + k.h / 2, k.fs, Paint.Align.CENTER)
        }
        w.btns = b
    }

    private fun key(c: String) {
        if (foc == "note" || foc == "ai") {
            var t = if (foc == "note") note else ai
            when (c) { "⌫" -> t = t.dropLast(1); "⏎" -> if (foc == "ai") { ai = t; askAI(); return } else t += "\n"; else -> { t += if (shift) c.uppercase() else c; shift = false } }
            if (foc == "note") note = t else ai = t
        }
    }

    private fun askAI() {
        val t = ai.trim(); if (t.isEmpty() || aiBusy) return
        if (apiKey.isEmpty()) { chat.add("! Gemini anahtarı yok (Ayarlar)"); return }
        chat.add("Sen: $t"); ai = ""; aiBusy = true
        Thread {
            var out: String
            try {
                val c = URL("https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent").openConnection() as HttpURLConnection
                c.requestMethod = "POST"; c.doOutput = true; c.connectTimeout = 15000; c.readTimeout = 30000
                c.setRequestProperty("Content-Type", "application/json"); c.setRequestProperty("x-goog-api-key", apiKey)
                val body = JSONObject().put("system_instruction", JSONObject().put("parts", JSONArray().put(JSONObject().put("text", "Türkçe, kısa ve net cevap ver."))))
                    .put("contents", JSONArray().put(JSONObject().put("role", "user").put("parts", JSONArray().put(JSONObject().put("text", t)))))
                c.outputStream.use { it.write(body.toString().toByteArray()) }
                val s = (if (c.responseCode < 400) c.inputStream else c.errorStream).bufferedReader().readText(); val j = JSONObject(s)
                out = if (j.has("candidates")) j.getJSONArray("candidates").getJSONObject(0).getJSONObject("content").getJSONArray("parts").getJSONObject(0).getString("text")
                else j.optJSONObject("error")?.optString("message") ?: "Cevap yok"
            } catch (e: Exception) { out = "Hata: ${e.message}" }
            chat.add("AI: $out"); aiBusy = false
        }.start()
    }

    // ---------------- El takibi + ışın (ray) etkileşimi ----------------
    private fun nv21(img: Image): ByteArray {
        val w = img.width; val h = img.height; val o = ByteArray(w * h * 3 / 2)
        val y = img.planes[0]; for (r in 0 until h) { y.buffer.position(r * y.rowStride); y.buffer.get(o, r * w, w) }
        val u = img.planes[1]; val v = img.planes[2]; var k = w * h
        for (r in 0 until h / 2) for (c in 0 until w / 2) { o[k++] = v.buffer.get(r * v.rowStride + c * v.pixelStride); o[k++] = u.buffer.get(r * u.rowStride + c * u.pixelStride) }
        return o
    }

    private fun detect(d: ByteArray, w: Int, h: Int) {
        try {
            val bo = ByteArrayOutputStream(); YuvImage(d, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), 85, bo)
            val bm = BitmapFactory.decodeByteArray(bo.toByteArray(), 0, bo.size())
            val res = lm?.detect(BitmapImageBuilder(bm).build()); val l = ArrayList<FloatArray>()
            res?.landmarks()?.forEach { p ->
                val a = p[8]; val b = p[4]
                val sz = hypot((p[0].x() - p[9].x()).toDouble(), (p[0].y() - p[9].y()).toDouble()).toFloat().coerceAtLeast(.02f)
                val pd = (hypot((a.x() - b.x()).toDouble(), (a.y() - b.y()).toDouble()) / sz).toFloat()
                l.add(floatArrayOf((a.x() + b.x()) / 2, (a.y() + b.y()) / 2, sz, pd))
            }
            raw = l; rawSeq++
        } catch (e: Exception) { } finally { busy = false }
    }

    private fun unproj(x: Float, y: Float, z: Float): FloatArray {
        val o = FloatArray(4); Matrix.multiplyMV(o, 0, inv, 0, floatArrayOf(x, y, z, 1f), 0); return floatArrayOf(o[0] / o[3], o[1] / o[3], o[2] / o[3])
    }

    private fun updateHands(frame: Frame) {
        if (!busy && lm != null) {
            try { val img = frame.acquireCameraImage(); try { val d = nv21(img); val w = img.width; val h = img.height; busy = true; ex.execute { detect(d, w, h) } } finally { img.close() } } catch (e: Exception) { }
        }
        val r = raw; val sq = rawSeq
        wins.forEach { it.hover = -1 }
        for ((i, h) in hands.withIndex()) {
            val m = r.getOrNull(i)
            if (sq != h.seq) {
                h.seq = sq
                if (m == null) { if (++h.miss > 5) { h.on = false; h.pinch = false; h.grab = null } }
                else {
                    h.miss = 0
                    val al = if (h.on) (.35f + hypot(m[0] - h.x, m[1] - h.y) * 12f).coerceIn(.35f, .95f) else 1f
                    h.x = if (h.on) h.x + (m[0] - h.x) * al else m[0]; h.y = if (h.on) h.y + (m[1] - h.y) * al else m[1]
                    h.sz = if (h.on) h.sz * .7f + m[2] * .3f else m[2]; h.on = true
                    val p = if (h.pinch) m[3] < .5f else m[3] < .3f
                    h.cnt = if (p) h.cnt + 1 else 0
                    val np = if (h.pinch) p else h.cnt >= 2
                    h.down = np && !h.pinch; h.pinch = np
                }
            }
            if (!h.on) continue
            val o = FloatArray(2); frame.transformCoordinates2d(Coordinates2d.IMAGE_NORMALIZED, floatArrayOf(h.x, h.y), Coordinates2d.VIEW_NORMALIZED, o)
            h.nx = o[0] * 2 - 1; h.ny = 1 - o[1] * 2
            val f = unproj(h.nx, h.ny, 1f); var dx = f[0] - cam[0]; var dy = f[1] - cam[1]; var dz = f[2] - cam[2]
            val dl = sqrt(dx * dx + dy * dy + dz * dz); dx /= dl; dy /= dl; dz /= dl
            var best: Win? = null; var bt = 1e9f; var bu = 0f; var bv = 0f
            for (w in wins) {
                val n = nrm(w); val rr = floatArrayOf(n[2], 0f, -n[0]); val den = dx * n[0] + dz * n[2]
                if (abs(den) < 1e-4f) continue
                val t = ((w.pos[0] - cam[0]) * n[0] + (w.pos[2] - cam[2]) * n[2]) / den
                if (t <= 0f || t >= bt) continue
                val px = cam[0] + dx * t - w.pos[0]; val py = cam[1] + dy * t - w.pos[1]; val pz = cam[2] + dz * t - w.pos[2]
                val u = (px * rr[0] + pz * rr[2]) / w.w + .5f; val v = .5f - py / w.hm
                if (u in 0f..1f && v in 0f..1f) { best = w; bt = t; bu = u * w.lw; bv = v * w.lh }
            }
            if (!h.pinch) h.grab = null
            val g = h.grab
            if (g != null) {
                val t = (h.t0 * h.sz0 / h.sz).coerceIn(.6f, 5f)
                g.pos = floatArrayOf(cam[0] + dx * t - h.off[0], cam[1] + dy * t - h.off[1], cam[2] + dz * t - h.off[2])
            } else if (best != null) {
                val bi = best.btns.indexOfFirst { bu >= it.x && bu <= it.x + it.w && bv >= it.y && bv <= it.y + it.h }
                best.hover = bi
                if (h.down) {
                    if (best.id == "note" || best.id == "ai") foc = best.id
                    if (bi >= 0) { vib(); best.btns[bi].fn() }
                    else if (best.id != "dock") { h.grab = best; h.t0 = bt; h.sz0 = h.sz
                        h.off[0] = cam[0] + dx * bt - best.pos[0]; h.off[1] = cam[1] + dy * bt - best.pos[1]; h.off[2] = cam[2] + dz * bt - best.pos[2] }
                }
            }
            h.down = false
        }
    }
}
