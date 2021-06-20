use jni::JNIEnv;

use jni::objects::{AutoArray, JClass, JList, JObject, JString};

use jni::objects::ReleaseMode;
use jni::sys::{
    jarray, jbyteArray, jdouble, jdoubleArray, jfloat, jfloatArray, jint, jintArray, jstring,
};
use real_hora;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;

trait ANNIndex:
    real_hora::core::ann_index::ANNIndex<f32, usize>
    + real_hora::core::ann_index::SerializableIndex<f32, usize>
{
}

pub fn metrics_transform(s: &str) -> real_hora::core::metrics::Metric {
    match s {
        "angular" => real_hora::core::metrics::Metric::Angular,
        "manhattan" => real_hora::core::metrics::Metric::Manhattan,
        "dot_product" => real_hora::core::metrics::Metric::DotProduct,
        "euclidean" => real_hora::core::metrics::Metric::Euclidean,
        "cosine_similarity" => real_hora::core::metrics::Metric::CosineSimilarity,
        _ => real_hora::core::metrics::Metric::Unknown,
    }
}

static mut ANNIndexManager: Option<
    HashMap<String, Box<real_hora::core::ann_index::ANNIndex<f32, usize>>>,
> = None;

#[no_mangle]
pub extern "system" fn Java_ANNIndexManager_init(env: JNIEnv, class: JClass) {
    unsafe {
        ANNIndexManager = Some(HashMap::<
            String,
            Box<real_hora::core::ann_index::ANNIndex<f32, usize>>,
        >::new());
    }
}

#[no_mangle]
pub extern "system" fn Java_ANNIndexManager_new_1bf_1index(
    env: JNIEnv,
    class: JClass,
    name: JString,
    dimension: jint,
) {
    let idx_name: String = env.get_string(name).unwrap().into();
    let idx_dimension = dimension as usize;
    unsafe {
        match &mut ANNIndexManager {
            Some(ann) => {
                &ann.insert(
                    idx_name,
                    Box::new(real_hora::index::bruteforce_idx::BruteForceIndex::<
                        f32,
                        usize,
                    >::new(
                        idx_dimension,
                        &real_hora::index::bruteforce_params::BruteForceParams::default(),
                    )),
                );
            }
            None => {}
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ANNIndexManager_add(
    env: JNIEnv,
    class: JClass,
    name: JString,
    features: jfloatArray,
    features_idx: jint,
) {
    let idx_name: String = env.get_string(name).unwrap().into();
    let idx = features_idx as usize;
    let length = env.get_array_length(features).unwrap() as usize;
    let mut buf: Vec<jfloat> = vec![0.0; length];
    env.get_float_array_region(features, 0, &mut buf).unwrap();

    unsafe {
        match &mut ANNIndexManager {
            Some(ann) => match ann.get_mut(&idx_name) {
                Some(index) => {
                    let n = real_hora::core::node::Node::new_with_idx(&buf, idx);
                    index.add_node(&n);
                }
                None => {}
            },
            None => {}
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ANNIndexManager_build(
    env: JNIEnv,
    class: JClass,
    name: JString,
    mt: JString,
) {
    let idx_name: String = env.get_string(name).unwrap().into();
    let metric: String = env.get_string(mt).unwrap().into();

    unsafe {
        match &mut ANNIndexManager {
            Some(ann) => match ann.get_mut(&idx_name) {
                Some(index) => {
                    index.build(metrics_transform(&metric));
                }
                None => {}
            },
            None => {}
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ANNIndexManager_search(
    env: JNIEnv,
    class: JClass,
    name: JString,
    k: jint,
    features: jfloatArray,
) -> jintArray {
    let idx_name: String = env.get_string(name).unwrap().into();
    let length = env.get_array_length(features).unwrap() as usize;
    let mut buf: Vec<jfloat> = vec![0.0; length];
    env.get_float_array_region(features, 0, &mut buf).unwrap();
    let topk = k as usize;
    let mut result: Vec<i32> = Vec::new();

    unsafe {
        match &mut ANNIndexManager {
            Some(ann) => match ann.get_mut(&idx_name) {
                Some(index) => {
                    result = index.search(&buf, topk).iter().map(|x| *x as i32).collect();
                }
                None => {}
            },
            None => {}
        }
    }

    let output = env.new_int_array(result.len() as i32).unwrap();
    env.set_int_array_region(output, 0, &result).unwrap();
    output
}
