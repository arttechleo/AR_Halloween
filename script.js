// Three.js from import map + GLTFLoader
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// ✅ MediaPipe Tasks (Pose) — ESM bundle with proper CORS/MIME
import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/vision_bundle.mjs";

/* -------------------- Config -------------------- */
const STATE = { running: false, landmarker: null, mixers: [], cameraMode: "user" };
const ASSETS = {
  shoulder: "./assets/model1.glb",
  back: "./assets/model2.glb",
  head: "./assets/model3.glb",
};
const CONF = {
  zVideo: -10, zModels: -5,
  poseEveryNFrames: 2,
  smooth: 0.6,
  shoulderOffsetMin: 0.38,
  backYOffset: 0.2,
  headScaleClamp: [0.6, 2.2]
};

/* -------------------- Globals -------------------- */
let video, overlay, octx, renderer, scene, camera, videoPlane, group;
let frame = 0;
let lastResult = null;

const IDX = { // MediaPipe pose indices
  NOSE: 0, LEFT_EYE: 2, RIGHT_EYE: 5, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12
};

/* -------------------- Three.js -------------------- */
function setupThree() {
  const threeEl = document.getElementById("three");
  const rect = threeEl.getBoundingClientRect();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(rect.width, rect.height);
  threeEl.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, rect.width / rect.height, 0.01, 100);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.0));

  const tex = new THREE.VideoTexture(video);
  tex.flipY = false;
  if (STATE.cameraMode === "user") { tex.wrapS = THREE.RepeatWrapping; tex.repeat.x = -1; tex.offset.x = 1; }

  const geo = new THREE.PlaneGeometry(1, 1); geo.scale(1, -1, 1);
  const mat = new THREE.MeshBasicMaterial({ map: tex, depthTest: false });
  videoPlane = new THREE.Mesh(geo, mat);
  scene.add(videoPlane); updateVideoPlaneSize();

  group = new THREE.Group(); scene.add(group);

  addEventListener("resize", () => {
    const r = threeEl.getBoundingClientRect();
    renderer.setSize(r.width, r.height);
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
    updateVideoPlaneSize();
  });
}
function updateVideoPlaneSize() {
  const threeEl = document.getElementById("three");
  const rect = threeEl.getBoundingClientRect();
  const aspect = rect.width / rect.height;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const h = Math.abs(2 * Math.tan(fov / 2) * CONF.zVideo);
  videoPlane.position.z = CONF.zVideo;
  videoPlane.scale.set(h * aspect, h, 1);
}

/* -------------------- GLB loader -------------------- */
function loadGLB(url) {
  return new Promise((res, rej) => {
    new GLTFLoader().load(
      url,
      (g) => {
        const root = g.scene || new THREE.Group();
        root.traverse(o => o.frustumCulled = false);
        if (g.animations?.length) {
          const mx = new THREE.AnimationMixer(root);
          g.animations.forEach(c => mx.clipAction(c).play());
          STATE.mixers.push(mx);
        }
        res(root);
      },
      undefined, rej
    );
  });
}

/* -------------------- Anchors -------------------- */
const anchors = { shoulderL: null, shoulderR: null, back: null, head: null };
async function setupAnchors() {
  const [m1, m2, m3] = await Promise.all([loadGLB(ASSETS.shoulder), loadGLB(ASSETS.back), loadGLB(ASSETS.head)]);
  anchors.shoulderL = m1.clone(true);
  anchors.shoulderR = m1.clone(true);
  anchors.back = m2;
  anchors.head = m3;
  [anchors.shoulderL, anchors.shoulderR, anchors.back, anchors.head].forEach(n => { n.visible = false; group.add(n); });
}

/* -------------------- Camera -------------------- */
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: STATE.cameraMode, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
  });
  video.srcObject = stream; await video.play();
}

/* -------------------- Pose (MediaPipe Tasks) -------------------- */
async function setupPose() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/wasm"
  );
  STATE.landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });
}

/* -------------------- Utils -------------------- */
const normX = (x) => x * 2 - 1;
const normY = (y) => -y * 2 + 1;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smoothSet = (obj, x, y, z) => {
  obj.position.x += (x - obj.position.x) * CONF.smooth;
  obj.position.y += (y - obj.position.y) * CONF.smooth;
  obj.position.z += (z - obj.position.z) * CONF.smooth;
};
function facePresent(lm) {
  const nose = lm[IDX.NOSE], le = lm[IDX.LEFT_EYE], re = lm[IDX.RIGHT_EYE];
  return !!(nose || le || re);
}

/* -------------------- Main loop -------------------- */
async function loop() {
  if (!STATE.running) return;
  requestAnimationFrame(loop);
  frame++;

  if (STATE.landmarker && video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA && frame % CONF.poseEveryNFrames === 0) {
    lastResult = STATE.landmarker.detectForVideo(video, performance.now());
  }

  const poses = lastResult?.landmarks;
  if (poses && poses.length) {
    const lm = poses[0]; // normalized [0..1]
    const lS = lm[IDX.LEFT_SHOULDER], rS = lm[IDX.RIGHT_SHOULDER];
    if (lS && rS) {
      const cx = (lS.x + rS.x) / 2, cy = (lS.y + rS.y) / 2;
      let gx = normX(cx), gy = normY(cy); if (STATE.cameraMode === "user") gx = -gx;
      group.position.set(gx, gy, CONF.zModels);

      const span = Math.abs(normX(lS.x) - normX(rS.x));
      const offset = Math.max(CONF.shoulderOffsetMin, span * 0.35);

      const hasFace = facePresent(lm);

      // shoulders (model1.glb) when face present
      [anchors.shoulderL, anchors.shoulderR].forEach(n => n.visible = !!hasFace);
      if (hasFace) {
        smoothSet(anchors.shoulderL,  offset, 0, 0);
        smoothSet(anchors.shoulderR, -offset, 0, 0);
        anchors.shoulderL.rotation.set(-0.2, Math.PI,  Math.PI * 0.08);
        anchors.shoulderR.rotation.set(-0.2, Math.PI, -Math.PI * 0.08);
        anchors.shoulderL.scale.setScalar(1.0);
        anchors.shoulderR.scale.setScalar(1.0);
      }

      // back (model2.glb) when no face
      anchors.back.visible = !hasFace;
      if (!hasFace) {
        smoothSet(anchors.back, 0, CONF.backYOffset, -0.2);
        anchors.back.rotation.set(0, Math.PI, 0);
        anchors.back.scale.setScalar(1.2);
      }

      // head (model3.glb) fully covering head when face present
      anchors.head.visible = hasFace;
      if (hasFace) {
        const nose = lm[IDX.NOSE] || { x: cx, y: cy - 0.04 };
        let nx = normX(nose.x), ny = normY(nose.y);
        if (STATE.cameraMode === "user") nx = -nx;

        const le = lm[IDX.LEFT_EYE], re = lm[IDX.RIGHT_EYE];
        const dx = (le && re) ? Math.abs(normX(le.x) - normX(re.x)) : span * 0.6;
        const headScale = clamp(dx * 2.2, CONF.headScaleClamp[0], CONF.headScaleClamp[1]);

        smoothSet(anchors.head, nx - group.position.x, ny - group.position.y + 0.1, -0.1);
        anchors.head.rotation.set(0, Math.PI, 0);
        anchors.head.scale.setScalar(headScale);
      }
    } else {
      Object.values(anchors).forEach(n => n && (n.visible = false));
    }
  }

  STATE.mixers.forEach(m => m.update(1 / 60));
  if (videoPlane?.material.map) videoPlane.material.map.needsUpdate = true;
  renderer.render(scene, camera);
}

/* -------------------- Boot -------------------- */
async function start() {
  document.getElementById("ui").style.display = "none";
  video = document.getElementById("video");
  overlay = document.getElementById("overlay");
  octx = overlay.getContext("2d");

  await setupCamera();
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;

  setupThree();
  await setupPose();
  await setupAnchors();

  STATE.running = true;
  loop();
}

addEventListener("DOMContentLoaded", () => {
  document.getElementById("start").addEventListener("click", start, { passive: true });
});
