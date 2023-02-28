import * as THREE from 'three';
// import BG from './environment/bg.jpg';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'


let scene, renderer, camera, stats;
let model, skeleton, mixer, clock;
let numAnimations;

var panel = new GUI( { width: 310 } );
const panelRegion = document.getElementById("model-panel")
panelRegion.style.boxSizing = "border-box"
panelRegion.style.textAlign = "center"
panel.domElement.style.position = "relative"
panel.domElement.style.margin = "10px 25px"
panel.domElement.style.boxShadow = "2px 2px 2px rgb(0 0 0 / 10%)"

// panel.domElement.style.padding = "10px"
panel.domElement.style.width = "97%"
panel.domElement.style.height = "80%"
panelRegion.appendChild(panel.domElement)


console.log(panel.domElement);
let folder1 = null;
let folder2 = null;
let folder3 = null;
let folder4 = null;
let folder5 = null;
let folder6 = null;

let panelSettings;

let singleStepMode = false;
let sizeOfNextStep = 0;

var allActions = [];

let sectionTitle = document.getElementById("model-generation-title")
const cards = document.getElementsByClassName("model-card")
const container = document.getElementById( 'model-generation-canvas' );

for (var i=0; i<cards.length; i++) {
    cards[i].onclick=function(){
        for (var j=0; j<cards.length; j++) {
            if (j!==i) {
                cards[j].style = ""
            }
        }
        var modelName = this.childNodes[2].childNodes[0].innerText
        sectionTitle.innerHTML = modelName
        modelName += "_idle"
        init(modelName)
        this.style = "-webkit-box-shadow: 0 0 6px #999; box-shadow: 0 0 6px #999; -webkit-transition: all .05s ease-out;transition: all .05s ease-out;"
    }
}

var defaultCard = document.getElementsByClassName("model-card")[0]
defaultCard.onclick()

collectUserInput()

function collectUserInput() {
    const btn= document.getElementById("user-input-btn");
    btn.addEventListener('click', function(){
        var motionName = document.getElementById("user-input-search").value;
        init(motionName)
    });
}

function init(modelName) {
    container.innerHTML = ""
    // collectUserInput()
    // scene
    clock = new THREE.Clock();
	scene = new THREE.Scene();
    scene.background = new THREE.TextureLoader().load('./environment/bg.jpeg');
	scene.fog = new THREE.Fog( 0xf5e1e1, 10, 50 );

    // lights
    const hemiLight = new THREE.HemisphereLight( 0xffffff, 0x444444 );
    hemiLight.position.set( 0, 20, 0 );
    scene.add( hemiLight );

    const dirLight = new THREE.DirectionalLight( 0xffffff );
    dirLight.position.set( 3, 10, 10 );
    dirLight.castShadow = true;
    dirLight.shadow.camera.top = 2;
    dirLight.shadow.camera.bottom = - 2;
    dirLight.shadow.camera.left = - 2;
    dirLight.shadow.camera.right = 2;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 40;
    scene.add( dirLight );

    // ground
    // var floorTexture = new THREE.ImageUtils.loadTexture( './imgs/checkerboard.jpg' );
    const floorTexture = new THREE.TextureLoader().load( './imgs/checkerboard.jpg' );  
    floorTexture.wrapS = floorTexture.wrapT = THREE.RepeatWrapping; 
	floorTexture.repeat.set( 500, 500 );

	// DoubleSide: render texture on both sides of mesh
	var floorMaterial = new THREE.MeshPhongMaterial( { color: 0x999999, map: floorTexture, side: THREE.DoubleSide} );
	var floorGeometry = new THREE.PlaneGeometry(1000, 1000, 1, 1);
	var floor = new THREE.Mesh(floorGeometry, floorMaterial);
	floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
	scene.add(floor);
    // scene.add( new THREE.CameraHelper( dirLight.shadow.camera ) );

    // ground
    // var plane = new THREE.PlaneGeometry( 100, 100, 100, 100 )
    // var meshBasic = new THREE.MeshPhongMaterial( {color: 0x999999, flatShading: true, vertexColors: true, shininess: 0} ) 
    // const wireframeMaterial = new THREE.MeshBasicMaterial( { color: 0x999999, wireframe: true, transparent: true } );

    // const count = plane.attributes.position.count;
    // const positionPlane = plane.attributes.position;

    // const color = new THREE.Color();
    // plane.setAttribute( 'color', new THREE.BufferAttribute( new Float32Array( count * 3 ), 3 ) );
    // const colorPlane = plane.attributes.color;

    // console.log(count)

    // for ( let i = 0; i < count; i ++ ) {
    //     var currentPosX = positionPlane.getX( i );
    //     var currentPosY = positionPlane.getY( i );
    //     // console.log("posX"+currentPosX)
    //     // console.log("posY"+currentPosY)
    //     if ((currentPosY == 0) && ((currentPosX == -1) || (currentPosX == 0) || (currentPosX == 1))) {
    //         color.setRGB( 255, 0, 0 );
    //         colorPlane.setXYZ( i, color.r, color.g, color.b );
    //     }
    // }

    // const mesh = new THREE.Mesh( plane, meshBasic );
    // const wireframe = new THREE.Mesh( plane, wireframeMaterial );
    // mesh.add( wireframe );

    // mesh.rotation.x = - Math.PI / 2;
    // mesh.receiveShadow = true;
    // scene.add( mesh );

    // model 
    modelLoader(modelName);

    // renderer
    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );

    var rendererWidth = document.getElementById("model-generation-canvas").clientWidth
    var rendererHeight = document.getElementById("model-generation-canvas").clientHeight
    
    // renderer.setSize( window.innerWidth/2, window.innerHeight/2 );
    renderer.setSize(rendererWidth, rendererWidth/2);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.shadowMap.enabled = true;
    renderer.domElement.style.boxShadow = "2px 2px 2px rgb(0 0 0 / 10%)";
    container.appendChild( renderer.domElement );

    // camera
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 100 );
    camera.position.set( -1, 2, 3 );

    // controls
    const controls = new OrbitControls( camera, renderer.domElement );
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.target.set( 0, 1, 0 );
    controls.update();
}

function modelLoader(modelName){
    // console.log(modelName)
    const loader = new GLTFLoader();
    const fbxLoader = new FBXLoader()
    
    if (modelName.endsWith('idle')) {
        modelName = modelName.slice(0, -5)
        var path = "../to_fbx/fbx_templates/" + modelName + "_Idle.fbx";	

        fbxLoader.load(
            path,
            function (object) {
                // object.traverse(function (child) {
                //     if ((child as THREE.Mesh).isMesh) {
                //         // (child as THREE.Mesh).material = material
                //         if ((child as THREE.Mesh).material) {
                //             ((child as THREE.Mesh).material as THREE.MeshBasicMaterial).transparent = false
                //         }
                //     }
                // })
                model = object
                model.scale.set(.011, .011, .011)
                // add model to scene
                scene.add(model);

                // console.log(model);

                model.traverse( function ( object ) {
                    if ( object.isMesh ) object.castShadow = true;
                } );

                // skeleton settings
                skeleton = new THREE.SkeletonHelper( model );
                skeleton.visible = false;
                scene.add(skeleton);

                // add animations
                const animations = model.animations;
                mixer = new THREE.AnimationMixer( model );
                numAnimations = animations.length;
                for ( let i = 0; i !== numAnimations; ++ i ) {

                    let clip = animations[ i ];
                    const name = clip.name;
                    const action = mixer.clipAction( clip );
                    activateAction( action );
                    allActions.push( action );
                }
                createPanel();
                animate();
        })
    } else {
        var currentModelName = sectionTitle.innerHTML
        
        // determine which motion
        if (modelName === "A person is walking") {
            var path = "../to_fbx/glb_models/" + currentModelName + ".glb"
        } else {
            var path = "../to_fbx/glb_models/" + currentModelName + ".glb"
        }

        loader.load( path, function ( gltf ) {
            model = gltf.scene;
            // add model to scene
            scene.add(model);
            model.traverse( function ( object ) {
                if ( object.isMesh ) object.castShadow = true;
            } );

            // skeleton settings
            skeleton = new THREE.SkeletonHelper( model );
            skeleton.visible = false;
            scene.add(skeleton);

            // add animations
            const animations = gltf.animations;
            mixer = new THREE.AnimationMixer( model );
            numAnimations = animations.length;
            for ( let i = 0; i !== numAnimations; ++ i ) {

                let clip = animations[ i ];
                const name = clip.name;
                const action = mixer.clipAction( clip );
                activateAction( action );
                allActions.push( action );
            }
            createPanel();

            animate();
        })
    }
}

function animate() {

    // Render loop
    requestAnimationFrame(animate);
    
    let mixerUpdateDelta = clock.getDelta();  

    if ( singleStepMode ) {

        mixerUpdateDelta = sizeOfNextStep;
        sizeOfNextStep = 0;

    }

    mixer.update( mixerUpdateDelta );
    renderer.render( scene, camera );
}

function activateAction( action ) {
    const clip = action.getClip();
    // const settings = baseActions[ clip.name ] || additiveActions[ clip.name ];
    setWeight( action, 1 );
    action.play();
}

function setWeight( action, weight ) {
    action.enabled = true;
    action.setEffectiveTimeScale( 1 );
    action.setEffectiveWeight( weight );
}

function createPanel() {

    if (folder2 !== null) {
        folder2.destroy();
        folder2 = null
    }

    if (folder3 !== null) {
        folder3.destroy();
        folder3 = null
    }

    if (folder4 !== null) {
        folder4.destroy();
        folder4 = null
    }

    folder2 = panel.addFolder( 'Pausing/Stepping' );
    folder3 = panel.addFolder( 'General Speed' );
    folder4 = panel.addFolder( 'Visibility' );
   
    panelSettings = {
        'show model': true,
        'show skeleton': false,
        'modify time scale': 1.0,
        'pause/continue': pauseContinue,
        'make single step': toSingleStepMode,
        'modify step size': 0.05,
    };

    folder2.add( panelSettings, 'pause/continue' );
    folder2.add( panelSettings, 'make single step' );
    folder2.add( panelSettings, 'modify step size', 0.01, 0.1, 0.001 );

    folder3.add( panelSettings, 'modify time scale', 0.0, 1.5, 0.01 ).onChange( modifyTimeScale );
    
    folder4.add( panelSettings, 'show model' ).onChange( showModel );
    folder4.add( panelSettings, 'show skeleton' ).onChange( showSkeleton );

    folder2.open();
    folder3.open();
    folder4.open();
}

function pauseContinue() {

    if ( singleStepMode ) {

        singleStepMode = false;
        unPauseAllActions();

    } else {

        if ( allActions[0].paused ) {

            unPauseAllActions();

        } else {

            pauseAllActions();

        }

    }

}

function toSingleStepMode() {
    unPauseAllActions();
    singleStepMode = true;
    sizeOfNextStep = panelSettings[ 'modify step size' ];
}

function pauseAllActions() {
    allActions.forEach( function ( action ) {
        action.paused = true;
    } );
}

function unPauseAllActions() {
    allActions.forEach( function ( action ) {
        action.paused = false;
    } );
}


function modifyTimeScale( speed ) {
    mixer.timeScale = speed;
}

function showModel( visibility ) {
    console.log(model)
    model.visible = visibility;
}


function showSkeleton( visibility ) {
    skeleton.visible = visibility;
}
