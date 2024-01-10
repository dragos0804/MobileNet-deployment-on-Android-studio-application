# Integrare de rețele neurale în aplicații Android
## Exemplificare folosind MobileNet

Rețeaua salvată sub format TFLite:
https://www.tensorflow.org/lite/examples/image_classification/overview

În cod, este necesară adăgarea path-ului către label-uri:

```kotlin
val labels = application.assets.open("labels.txt").bufferedReader().use { it.readText() }.split("\n")
```

Adăugarea de ependențe:

```kotlin
dependencies {
    ...
implementation 'org.tensorflow:tensorflow-lite-support:0.1.0-rc1'
implementation 'org.tensorflow:tensorflow-lite-metadata:0.1.0-rc1'
    ...
}
```

Butonul cu ID make_prediction, la click, instanțiază modelul, preia feature-urile și salvează pe output rezultatele predicției, urmând să preia cea mai mare probabilitate din distribuție:

```kotlin
make_prediction.setOnClickListener(View.OnClickListener {
            var resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val model = MobilenetV110224Quant.newInstance(this)

            var tbuffer = TensorImage.fromBitmap(resized)
            var byteBuffer = tbuffer.buffer

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            var max = getMax(outputFeature0.floatArray)

            text_view.setText(labels[max])

// Releases model resources if no longer used.
            model.close()
        })
```