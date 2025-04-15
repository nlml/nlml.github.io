console.log("Loading meshes.js");



class MeshScene {
    constructor(containerId, meshColor, vertices) {
        this.container = document.getElementById(containerId);
        this.meshColor = meshColor;
        this.vertices = vertices;

        this.init();
    }

    init() {
        // Get container dimensions
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        // Create scene
        this.scene = new THREE.Scene();

        // Create camera
        this.camera = new THREE.PerspectiveCamera(14.23, width / height, 0.01, 5.0);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setClearColor(0x222222); // Dark gray background
        this.container.appendChild(this.renderer.domElement);

        // Create controls
        this.controls = new SimpleOrbitControls(this.camera, this.renderer.domElement);

        // Create the mesh
        this.createMesh(this.vertices, faces);

        // Add lights
        this.addLights();

        // Handle window resize
        window.addEventListener('resize', this.onResize.bind(this));

        // Start animation
        this.animate();
    }


    createMesh(vertices, faces) {
        // Create a buffer geometry
        const geometry = new THREE.BufferGeometry();

        // Divide all vertices by 4.0
        const scale = 1.0; /// 4.0;  // TODO(LS): rm this
        vertices = vertices.map(vertex => vertex.map(coord => coord * scale));

        // Create position array from vertices
        const positionArray = [];
        vertices.forEach(vertex => {
          positionArray.push(...vertex); // Spread the x, y, z coordinates
        });

        // Set position attribute (vertices)
        geometry.setAttribute(
          'position',
          new THREE.Float32BufferAttribute(positionArray, 3)
        );

        // Set indices for faces
        const indices = [];
        faces.forEach(face => {
          indices.push(...face); // Spread the vertex indices of each face
        });

        // Add faces as indices
        geometry.setIndex(indices);

        // Compute normals
        geometry.computeVertexNormals();

        // Create material
        const material = new THREE.MeshPhongMaterial({
            specular: 0x111111,
            shininess: 30,
            flatShading: false,
        });

        // Create and add mesh
        this.mesh = new THREE.Mesh(geometry, material);
        this.scene.add(this.mesh);
      }

    addLights() {
        const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
        light1.position.set(-3, 3, 3);
        this.scene.add(light1);

        const light2 = new THREE.DirectionalLight(0xffffff, 0.3);
        light2.position.set(3, -3, -3);
        this.scene.add(light2);

        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.renderer.setSize(width, height);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));

        // Auto-rotate the mesh
        // this.mesh.rotation.x += 0.003;
        // this.mesh.rotation.y += 0.005;

        // Update controls
        this.controls.update();

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        // Clean up resources when no longer needed
        this.controls.dispose();
        window.removeEventListener('resize', this.onResize);
        this.container.removeChild(this.renderer.domElement);
    }
}

// Initialize viewers for each gallery item
// domContentLoaded
document.addEventListener("DOMContentLoaded", function() {
    const meshScene1 = new MeshScene('viewer1', 0xffffff, vertices1); // Red
    const meshScene2 = new MeshScene('viewer2', 0xffffff, vertices2); // Red
    const meshScene3 = new MeshScene('viewer3', 0xffffff, vertices3); // Red
    const meshScene4 = new MeshScene('viewer4', 0xffffff, vertices4); // Red
    const meshScene5 = new MeshScene('viewer5', 0xffffff, vertices5); // Red
});
