#![allow(unused_imports)]

use windows::{AI::MachineLearning::*,
              Foundation::Collections,
              Graphics::Imaging,
              Media::VideoFrame,
              Globalization::Language,
              Media::Ocr::*,
              Storage,
              Win32};
use windows::core::*;
use windows::Graphics::Imaging::SoftwareBitmap;


type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub enum Device {
    Default,
    Cpu,
    DirectX,
    DirectXHighPerformance,
    DirectXMinPower,
}

pub struct OnnxModelSession {
    pub model_path: String,
    pub device: LearningModelDevice,
    pub model: LearningModel,
    pub session: LearningModelSession,
}

impl OnnxModelSession {
    pub fn build(model_path: &str, device: Device) -> Res<Self> {
        let model_path_h = HSTRING::from(model_path);
        let device_kind = match device {
            Device::Default => LearningModelDeviceKind::Default,
            Device::Cpu => LearningModelDeviceKind::Cpu,
            Device::DirectX => LearningModelDeviceKind::DirectX,
            Device::DirectXHighPerformance => LearningModelDeviceKind::DirectXHighPerformance,
            Device::DirectXMinPower => LearningModelDeviceKind::DirectXMinPower
        };
        let device = LearningModelDevice::Create(device_kind)?;
        let model = LearningModel::LoadFromFilePath(&model_path_h)?;
        let session = LearningModelSession::CreateFromModelOnDevice(&model, &device)?;

        Ok(OnnxModelSession {
            model_path: model_path.to_string(),
            device,
            model,
            session
            ,
        })
    }

    pub fn inspect(&self) -> Res<()> {
        println!("Model \x1B[031m{}\x1B[0m in \x1B[035m{}\x1B[0m", self.model.Name()?, self.model_path);
        println!("\x1B[033mVersion\x1B[0m: \x1B[032m{}\x1B[0m\n\x1B[033mAuthor\x1B[0m: \x1B[032m{}\x1B[0m\n\x1B[033mDomain: \x1B[033m{}\x1B[0m\n\x1B[033mDescription\x1B[0m: {}\n",
                 self.model.Version()?,
                 self.model.Author()?,
                 self.model.Domain()?,
                 self.model.Description()?
        );
        let meta = self.model.Metadata()?;
        for m in meta {
            println!("{:?}:{:?}", m.Key()?, m.Value()?)
        }
        let input = self.model.InputFeatures()?;
        let output = self.model.OutputFeatures()?;
        for i in input {
            let kind = i.Kind()?;
            if kind == LearningModelFeatureKind::Tensor {
                let tensor_descriptor: TensorFeatureDescriptor = i.cast()?;
                let tensor_kind = match tensor_descriptor.TensorKind()? {
                    TensorKind::Undefined => "Undefined",
                    TensorKind::UInt8 => "U8",
                    TensorKind::Boolean => "Bool",
                    TensorKind::String => "String",
                    TensorKind::UInt16 => "U16",
                    TensorKind::UInt32 => "U32",
                    TensorKind::UInt64 => "U64",
                    TensorKind::Int8 => "I8",
                    TensorKind::Int16 => "I16",
                    TensorKind::Int32 => "I32",
                    TensorKind::Int64 => "I64",
                    TensorKind::Float => "F32",
                    TensorKind::Float16 => "F16",
                    TensorKind::Double => "F64",
                    TensorKind::Complex64 => "C64",
                    TensorKind::Complex128 => "C128",
                    _ => "Unknown"
                };
                let shape: Vec<_> = tensor_descriptor.Shape()?.into_iter().collect();
                let require = tensor_descriptor.IsRequired()?;
                println!("\x1B[033mInput:\x1B[0m \x1B[031m{:}\x1B[0m{:}:\x1B[036m{:}\x1B[0m \x1B[032m{:?}\x1B[0m \x1B[034m{:}\x1B[0m {:} ", i.Name()?, if require { "" } else { "*" }, "Tensor", shape, tensor_kind, i.Description()?);
            }
        }
        for i in output {
            let kind = i.Kind()?;
            if kind == LearningModelFeatureKind::Tensor {
                let tensor_descriptor: TensorFeatureDescriptor = i.cast()?;
                let tensor_kind = match tensor_descriptor.TensorKind()? {
                    TensorKind::Undefined => "Undefined",
                    TensorKind::UInt8 => "U8",
                    TensorKind::Boolean => "Bool",
                    TensorKind::String => "String",
                    TensorKind::UInt16 => "U16",
                    TensorKind::UInt32 => "U32",
                    TensorKind::UInt64 => "U64",
                    TensorKind::Int8 => "I8",
                    TensorKind::Int16 => "I16",
                    TensorKind::Int32 => "I32",
                    TensorKind::Int64 => "I64",
                    TensorKind::Float => "F32",
                    TensorKind::Float16 => "F16",
                    TensorKind::Double => "F64",
                    TensorKind::Complex64 => "C64",
                    TensorKind::Complex128 => "C128",
                    _ => "Unknown"
                };
                let shape: Vec<_> = tensor_descriptor.Shape()?.into_iter().collect();
                let require = tensor_descriptor.IsRequired()?;
                println!("\x1B[033mOutput:\x1B[0m \x1B[031m{:}\x1B[0m{:}:\x1B[036m{:}\x1B[0m \x1B[032m{:?}\x1B[0m \x1B[034m{:}\x1B[0m {:} ", i.Name()?, if require { "" } else { "*" }, "Tensor", shape, tensor_kind, i.Description()?);
            }
        }
        Ok(())
    }

    pub fn predict_h<T: FnOnce(&LearningModelBinding)>(&self, handle: T) -> Res<LearningModelEvaluationResult> {
        let binding = LearningModelBinding::CreateFromSession(&self.session)?;
        handle(&binding);
        let res = self.session.Evaluate(&binding, &HSTRING::from("run_id"))?;
        Ok(res)
    }
    // pub fn predict(&self,input_name:&str){
    //     let binding = LearningModelBinding::CreateFromSession(&self.session)?;
    //
    // }
}


pub fn load_img_as_bitmap(path: &str) -> Res<SoftwareBitmap> {
    let file = Storage::StorageFile::GetFileFromPathAsync(&HSTRING::from(path))?.get()?;
    let stream = file.OpenAsync(Storage::FileAccessMode::Read)?.get()?;
    let decoder = Imaging::BitmapDecoder::CreateAsync(&stream)?.get()?;
    let software_bitmap = decoder.GetSoftwareBitmapAsync()?.get()?;
    Ok(software_bitmap)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_inspect() {
        let model = OnnxModelSession::build(r#"D:\绘世启动器\.ext\Lib\site-packages\onnx\backend\test\data\simple\test_sequence_model3\model.onnx"#, Device::Default).unwrap();
        model.inspect().unwrap()
    }
}