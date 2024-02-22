use windows::{AI::MachineLearning::*,
              Foundation::Collections,
              Graphics::Imaging,
              Media::VideoFrame,
              Globalization::Language,
              Media::Ocr::*,
              Storage,
              Win32};
use windows::core::*;
use std::thread::sleep;
use std::time::Duration;

pub fn ml() -> Result<()> {
    let start = std::time::Instant::now();

    let model_path = HSTRING::from("SqueezeNet.onnx");
    let img_path = HSTRING::from(r#"E:\WorkSpace\RustProjects\winml-rs\test.png"#);
    let device_name = "default";
    let device_kind = LearningModelDeviceKind::DirectXHighPerformance;
    let device = LearningModelDevice::Create(device_kind)?;



    // load the model
    println!("Loading model file {} on the {} device\n", model_path, device_name);
    let model = LearningModel::LoadFromFilePath(&model_path)?;
    let session = LearningModelSession::CreateFromModelOnDevice(&model, &device)?;


    let file = Storage::StorageFile::GetFileFromPathAsync(&img_path)?.get()?;
    let stream = file.OpenAsync(Storage::FileAccessMode::Read)?.get()?;
    let decoder = Imaging::BitmapDecoder::CreateAsync(&stream)?.get()?;
    let software_bitmap = decoder.GetSoftwareBitmapAsync()?.get()?;
    let input_image = VideoFrame::CreateWithSoftwareBitmap(&software_bitmap)?;




    for _ in 0..1000 {
        let file = Storage::StorageFile::GetFileFromPathAsync(&img_path)?.get()?;
        let stream = file.OpenAsync(Storage::FileAccessMode::Read)?.get()?;
        let decoder = Imaging::BitmapDecoder::CreateAsync(&stream)?.get()?;
        let software_bitmap = decoder.GetSoftwareBitmapAsync()?.get()?;
        let input_image = VideoFrame::CreateWithSoftwareBitmap(&software_bitmap)?;

        let binding = LearningModelBinding::CreateFromSession(&session)?;

        binding.Bind(&HSTRING::from("data_0"), &ImageFeatureValue::CreateFromVideoFrame(&input_image)?)?;

        let shape = vec![1i64, 1000, 1, 1];
        binding.Bind(&HSTRING::from("softmaxout_1"), &TensorFloat::Create2(&Collections::IVectorView::<i64>::try_from(shape)?)?)?;
        // binding.Bind(&HSTRING::from("softmaxout_1"), &TensorFloat::CreateFromShapeArrayAndDataArray(&[4, ][..], &[1f32, 1000f32, 1f32, 1f32][..])?)?;



        let result = session.Evaluate(&binding, &HSTRING::from("run_id"))?;
        let res_tensor: TensorFloat = result.Outputs()?.Lookup(&HSTRING::from("softmaxout_1"))?.cast()?;
        let mut res_vec: Vec<_> = res_tensor
            .GetAsVectorView()?
            .into_iter()
            .collect();
        let max = res_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        println!("res:{} {}", max.0, max.1);

    }


    println!("{:?}", start.elapsed());

    Ok(())
}
// use windows::{core::Result, Win32::System::Threading::*};

pub fn ocr() -> Result<()> {
    let start = std::time::Instant::now();

    let img_path = HSTRING::from(r#"E:\WorkSpace\RustProjects\winml-rs\test1.png"#);


    let language = Language::CreateLanguage(&HSTRING::from("zh-Hans"))?;
    let engine = OcrEngine::TryCreateFromLanguage(&language)?;

    for _ in 0..100 {
        let file1 = Storage::StorageFile::GetFileFromPathAsync(&img_path)?;
        let file=file1.get()?;

        let stream = file.OpenAsync(Storage::FileAccessMode::Read)?.get()?;
        let decoder = Imaging::BitmapDecoder::CreateAsync(&stream)?.get()?;
        let software_bitmap = decoder.GetSoftwareBitmapAsync()?.get()?;

        let result = engine.RecognizeAsync(&software_bitmap)?.get()?;

        let lines = result.Lines()?;

        println!("{:?}",lines.Size());
        for line in lines {
            println!("{}", line.Text().unwrap());
        }
    }
    println!("{:?}", start.elapsed());
    // let _ = lines
    //     .into_iter()
    //     .map(|x| println!("{}", x.Text().unwrap()));


    Ok(())
}