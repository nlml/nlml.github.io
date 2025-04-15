/**
 * Simple orbit controls implementation for Three.js
 */
class SimpleOrbitControls {
    constructor(camera, domElement) {
        this.camera = camera;
        this.domElement = domElement;

        this.rotateX = 0;
        this.rotateY = 0;
        this.zoom = 1;
        this.isDragging = false;
        this.previousMousePosition = { x: 0, y: 0 };

        // Bind methods
        this.onMouseDown = this.onMouseDown.bind(this);
        this.onMouseUp = this.onMouseUp.bind(this);
        this.onMouseMove = this.onMouseMove.bind(this);
        this.onWheel = this.onWheel.bind(this);

        // Add event listeners
        this.addEventListeners();
    }

    addEventListeners() {
        this.domElement.addEventListener('mousedown', this.onMouseDown);
        document.addEventListener('mouseup', this.onMouseUp);
        document.addEventListener('mousemove', this.onMouseMove);
        this.domElement.addEventListener('wheel', this.onWheel, { passive: false });
    }

    removeEventListeners() {
        this.domElement.removeEventListener('mousedown', this.onMouseDown);
        document.removeEventListener('mouseup', this.onMouseUp);
        document.removeEventListener('mousemove', this.onMouseMove);
        this.domElement.removeEventListener('wheel', this.onWheel);
    }

    onMouseDown(e) {
        const rect = this.domElement.getBoundingClientRect();
        if (e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top && e.clientY <= rect.bottom) {
            this.isDragging = true;
            this.previousMousePosition = {
                x: e.clientX,
                y: e.clientY
            };
        }
    }

    onMouseUp() {
        this.isDragging = false;
    }

    onMouseMove(e) {
        if (this.isDragging) {
            const deltaMove = {
                x: e.clientX - this.previousMousePosition.x,
                y: e.clientY - this.previousMousePosition.y
            };

            this.rotateY += deltaMove.x * 0.01;
            this.rotateX += deltaMove.y * 0.01;

            this.previousMousePosition = {
                x: e.clientX,
                y: e.clientY
            };
        }
    }

    onWheel(e) {
        const rect = this.domElement.getBoundingClientRect();
        if (e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top && e.clientY <= rect.bottom) {
            this.zoom += e.deltaY * 0.001;
            this.zoom = Math.max(1, Math.min(this.zoom, 10)); // Limit zoom between 1 and 10
            e.preventDefault();
        }
    }

    update() {
        // Update camera position based on current rotation and zoom
        this.camera.position.x = Math.sin(this.rotateY) * this.zoom;
        this.camera.position.z = Math.cos(this.rotateY) * this.zoom;
        this.camera.position.y = Math.sin(this.rotateX) * this.zoom;

        this.camera.lookAt(0, 0, 0);
    }

    dispose() {
        this.removeEventListeners();
    }
}
