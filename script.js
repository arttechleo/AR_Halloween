import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/vision_bundle.mjs";

const ASSETS = {
  model1: "./assets/model1.glb", // shoulders when face detected
  model2: "./assets/model2.glb", // back when no face
  model3: "./assets/model3.glb", // head cover when face detected
};

const CONF = {
  zVideo: -10, zModels: -5, poseEveryN: 2,
  smooth: 0.6, shoulderOffsetMin: 0.38, backYOffset: 0.2,
  headScaleClamp: [0.6, 2.2]
};

const IDX = { NOSE:0, LEFT_EYE:2, RIGHT_EYE:5, LEFT_SHOULDER:11, RIGHT_SHOULDER:12 };

const STATE = {
  running:false, inited:false, anchorsReady:false,
  landmarker:null, mixers:[], stream:null,
  cameraMode: /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ? "environment" : "user"
};

const DOM = { video:null, overlay:null, ui:null, startBtn:null, three:null };
const $ = s => document.querySelector(s);

function ensureDom(){
  DOM.video   = $("#video");
  DOM.overlay = $("#overlay");
  DOM.ui      = $("#overlay-ui");
  DOM.startBtn= $("#start");
  DOM.three   = $("#three");
  if(!DOM.video||!DOM.overlay||!DOM.ui||!DOM.startBtn||!DOM.three) throw new Error("Missing DOM elements");
}

const normX = x => x*2-1;
const normY = y => -y*2+1;
const clamp = (v,a,b)=>Math.min(b,Math.max(a,v));
const facePresent = lm => !!(lm[IDX.NOSE]||lm[IDX.LEFT_EYE]||lm[IDX.RIGHT_EYE]);
const smoothSet = (o,x,y,z)=>{ o.position.x+=(x-o.position.x)*CONF.smooth; o.position.y+=(y-o.position.y)*CONF.smooth; o.position.z+=(z-o.position.z)*CONF.smooth; };

// THREE setup
let renderer, scene, camera, videoPlane, group, frame=0, lastResult=null;

function setupThree(){
  if (renderer) return;
  const r = DOM.three.getBoundingClientRect();
  renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true, powerPreference:"high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(r.width, r.height);
  DOM.three.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, r.width/r.height, 0.01, 100);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.0));

  const tex = new THREE.VideoTexture(DOM.video);
  tex.flipY = false;
  if (STATE.cameraMode === "user") { tex.wrapS = THREE.RepeatWrapping; tex.repeat.x = -1; tex.offset.x = 1; }
  const geo = new THREE.PlaneGeometry(1,1); geo.scale(1,-1,1);
  const mat = new THREE.MeshBasicMaterial({ map: tex, depthTest:false });
  videoPlane = new THREE.Mesh(geo, mat);
  scene.add(videoPlane); updateVideoPlaneSize();

  group = new THREE.Group(); scene.add(group);

  addEventListener("resize", ()=>{
    const rr = DOM.three.getBoundingClientRect();
    renderer.setSize(rr.width, rr.height);
    camera.aspect = rr.width/rr.height;
    camera.updateProjectionMatrix();
    updateVideoPlaneSize();
  }, { passive:true });
}

function updateVideoPlaneSize(){
  const rect = DOM.three.getBoundingClientRect();
  const aspect = rect.width/rect.height;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const h = Math.abs(2*Math.tan(fov/2)*CONF.zVideo);
  videoPlane.position.z = CONF.zVideo;
  videoPlane.scale.set(h*aspect, h, 1);
}

// GLB loader
function loadGLB(url, label){
  return new Promise((resolve)=>{
    new GLTFLoader().load(
      url,
      (g)=>{
        const root = g.scene || new THREE.Group();
        root.traverse(o=>o.frustumCulled=false);
        if (g.animations?.length){
          const mx = new THREE.AnimationMixer(root);
          g.animations.forEach(c=>mx.clipAction(c).play());
          STATE.mixers.push(mx);
        }
        resolve(root);
      },
      undefined,
      ()=>{
        const box = new THREE.Mesh(new THREE.BoxGeometry(0.3,0.3,0.3), new THREE.MeshNormalMaterial());
        box.userData.fallback = label; resolve(box);
      }
    );
  });
}

// Anchors
const anchors = { shoulderL:null, shoulderR:null, back:null, head:null };
async function setupAnchors(){
  if (STATE.anchorsReady) return;
  const [m1,m2,m3] = await Promise.all([
    loadGLB(ASSETS.model1,"model1"),
    loadGLB(ASSETS.model2,"model2"),
    loadGLB(ASSETS.model3,"model3")
  ]);
  anchors.shoulderL = m1.clone(true);
  anchors.shoulderR = m1.clone(true);
  anchors.back = m2; anchors.head = m3;
  [anchors.shoulderL, anchors.shoulderR, anchors.back, anchors.head].forEach(n=>{ n.visible=false; group.add(n); });
  STATE.anchorsReady = true;
}

// Camera
async function getCameraStream(){
  const tryC = async (c)=>{ try{ return await navigator.mediaDevices.getUserMedia(c); }catch(e){ if(e.name==="OverconstrainedError"||e.name==="NotFoundError") return null; throw e; } };
  let s = await tryC({ video:{ facingMode:{ exact:"environment" } }, audio:false });
  if (s) return s;
  s = await tryC({ video:{ facingMode:"environment" }, audio:false });
  if (s) return s;
  s = await tryC({ video:true, audio:false });
  return s;
}

async function acquireCamera(){
  if (!isSecureContext) throw new Error("Site is not HTTPS.");
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API unavailable.");
  const stream = await getCameraStream();
  if (!stream) throw new Error("No usable camera found.");
  STATE.stream = stream;
  DOM.video.srcObject = stream;
  await DOM.video.play();
}

// MediaPipe
async function setupPose(){
  if (STATE.landmarker) return;
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/wasm");
  STATE.landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" },
    runningMode:"VIDEO", numPoses:1
  });
}

// Loop
async function loop(){
  if (!STATE.running) return;
  requestAnimationFrame(loop);
  frame++;

  if (STATE.landmarker && DOM.video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA && frame % CONF.poseEveryN === 0){
    lastResult = STATE.landmarker.detectForVideo(DOM.video, performance.now());
  }

  const poses = lastResult?.landmarks;
  if (poses && poses.length){
    const lm = poses[0];
    const lS = lm[IDX.LEFT_SHOULDER], rS = lm[IDX.RIGHT_SHOULDER];
    if (lS && rS){
      const cx = (lS.x+rS.x)/2, cy = (lS.y+rS.y)/2;
      let gx = normX(cx), gy = normY(cy); if (STATE.cameraMode === "user") gx = -gx;
      group.position.set(gx, gy, CONF.zModels);

      const span = Math.abs(normX(lS.x) - normX(rS.x));
      const offset = Math.max(CONF.shoulderOffsetMin, span*0.35);
      const hasFace = facePresent(lm);

      [anchors.shoulderL, anchors.shoulderR].forEach(n=> n.visible = !!hasFace);
      if (hasFace){
        smoothSet(anchors.shoulderL,  offset, 0, 0);
        smoothSet(anchors.shoulderR, -offset, 0, 0);
        anchors.shoulderL.rotation.set(-0.2, Math.PI,  Math.PI*0.08);
        anchors.shoulderR.rotation.set(-0.2, Math.PI, -Math.PI*0.08);
        anchors.shoulderL.scale.setScalar(1.0);
        anchors.shoulderR.scale.setScalar(1.0);
      }

      anchors.back.visible = !hasFace;
      if (!hasFace){
        smoothSet(anchors.back, 0, CONF.backYOffset, -0.2);
        anchors.back.rotation.set(0, Math.PI, 0);
        anchors.back.scale.setScalar(1.2);
      }

      anchors.head.visible = hasFace;
      if (hasFace){
        const nose = lm[IDX.NOSE] || { x: cx, y: cy - 0.04 };
        let nx = normX(nose.x), ny = normY(nose.y); if (STATE.cameraMode === "user") nx = -nx;
        const le = lm[IDX.LEFT_EYE], re = lm[IDX.RIGHT_EYE];
        const dx = (le && re) ? Math.abs(normX(le.x)-normX(re.x)) : span*0.6;
        const headScale = clamp(dx*2.2, CONF.headScaleClamp[0], CONF.headScaleClamp[1]);
        smoothSet(anchors.head, nx - group.position.x, ny - group.position.y + 0.1, -0.1);
        anchors.head.rotation.set(0, Math.PI, 0);
        anchors.head.scale.setScalar(headScale);
      }
    } else {
      Object.values(anchors).forEach(n=> n && (n.visible=false));
    }
  }

  STATE.mixers.forEach(m=>m.update(1/60));
  if (videoPlane?.material.map) videoPlane.material.map.needsUpdate = true;
  renderer.render(scene, camera);
}

// Start
async function start(){
  if (STATE.inited) return;
  STATE.inited = true;
  DOM.err.textContent = "";
  DOM.ui.style.display = "none";

  try {
    await acquireCamera();
  } catch (e){
    DOM.ui.style.display = "flex";
    DOM.err.textContent = e.message || "Camera blocked.";
    STATE.inited = false;
    return;
  }

  DOM.overlay.width  = DOM.video.videoWidth  || 1280;
  DOM.overlay.height = DOM.video.videoHeight || 720;

  setupThree();
  await setupPose();
  await setupAnchors();

  STATE.running = true;
  loop();
}

// UI
window.addEventListener("DOMContentLoaded", () => {
  try { ensureDom(); } catch(e){ alert(e.message); return; }
  DOM.startBtn.addEventListener("click", start, { passive:true });
});
