package com.example.detector

import android.graphics.*
import android.os.Bundle
import android.widget.ImageView
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {

    private val modelPath = "best_float16.tflite"
    private val labelPath = "labels.txt"
    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STD))
        .add(CastOp(INPUT_TYPE))
        .build()

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STD = 255f
        private val INPUT_TYPE = org.tensorflow.lite.DataType.FLOAT32
        private val OUTPUT_TYPE = org.tensorflow.lite.DataType.FLOAT32
        private const val CONF_THRESHOLD = 0.3f
        private const val IOU_THRESHOLD = 0.5f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btnPredict = findViewById<Button>(R.id.btn_predict)
        val imageView = findViewById<ImageView>(R.id.imageView)

        // Mostrar imagen antes de presionar predecir
        val inputStream = assets.open("image4.jpg")
        val originalBitmap = BitmapFactory.decodeStream(inputStream)
        imageView.setImageBitmap(originalBitmap)


        // Inicializar modelo
        val model = FileUtil.loadMappedFile(this, modelPath)
        val options = Interpreter.Options().apply { numThreads = 4 }
        interpreter = Interpreter(model, options)

        val inputShape = interpreter!!.getInputTensor(0).shape()
        val outputShape = interpreter!!.getOutputTensor(0).shape()
        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]

        // Cargar etiquetas
        val reader = BufferedReader(InputStreamReader(assets.open(labelPath)))
        reader.forEachLine { line -> if (line.isNotEmpty()) labels.add(line) }
        reader.close()

        btnPredict.setOnClickListener {
            // Cargar imagen desde assets
            val inputStream = assets.open("image4.jpg")
            val originalBitmap = BitmapFactory.decodeStream(inputStream)

            val resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, tensorWidth, tensorHeight, false)
            val tensorImage = TensorImage(INPUT_TYPE)
            tensorImage.load(resizedBitmap)
            val processedImage = imageProcessor.process(tensorImage)
            val imageBuffer = processedImage.buffer

            val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_TYPE)
            interpreter!!.run(imageBuffer, output.buffer)

            val boxes = bestBox(output.floatArray)
            val resultBitmap = drawBoundingBoxes(originalBitmap, boxes ?: emptyList())

            // Mostrar imagen con cajas en el ImageView
            imageView.setImageBitmap(resultBitmap)
        }
    }

    data class BoundingBox(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val cx: Float, val cy: Float, val w: Float, val h: Float,
        val cnf: Float, val cls: Int, val clsName: String
    )

    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boxes = mutableListOf<BoundingBox>()
        for (c in 0 until numElements) {
            var maxConf = -1f
            var maxIdx = -1
            var j = 4
            var idx = c + numElements * j
            while (j < numChannel) {
                if (array[idx] > maxConf) {
                    maxConf = array[idx]
                    maxIdx = j - 4
                }
                j++
                idx += numElements
            }

            if (maxConf > CONF_THRESHOLD) {
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - w / 2f
                val y1 = cy - h / 2f
                val x2 = cx + w / 2f
                val y2 = cy + h / 2f

                if (x1 < 0f || y1 < 0f || x2 > 1f || y2 > 1f) continue

                boxes.add(BoundingBox(x1, y1, x2, y2, cx, cy, w, h, maxConf, maxIdx, labels[maxIdx]))
            }
        }

        return if (boxes.isEmpty()) null else applyNMS(boxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): MutableList<BoundingBox> {
        val sorted = boxes.sortedByDescending { it.cnf }.toMutableList()
        val result = mutableListOf<BoundingBox>()

        while (sorted.isNotEmpty()) {
            val first = sorted.removeAt(0)
            result.add(first)

            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val box = iterator.next()
                if (calculateIoU(first, box) > IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return result
    }

    private fun calculateIoU(a: BoundingBox, b: BoundingBox): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)

        val interArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val areaA = a.w * a.h
        val areaB = b.w * b.h

        return interArea / (areaA + areaB - interArea)
    }

    private fun drawBoundingBoxes(bitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 6f
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 40f
            typeface = Typeface.DEFAULT_BOLD
        }

        for (box in boxes) {
            val rect = RectF(
                box.x1 * bitmap.width,
                box.y1 * bitmap.height,
                box.x2 * bitmap.width,
                box.y2 * bitmap.height
            )
            canvas.drawRect(rect, paint)
            canvas.drawText(box.clsName, rect.left, rect.top - 10f, textPaint)
        }

        return mutableBitmap
    }
}
