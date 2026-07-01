# Shader & 3D Reference

## When to Use What

| Need | Library | Weight |
|------|---------|--------|
| Background ambiance (gradients, noise, liquid) | ogl | ~15KB |
| Particle systems (simple) | Canvas API | 0KB (native) |
| Particle systems (complex, GPU) | Three.js | ~150KB |
| Interactive 3D objects | Three.js + @react-three/fiber | ~150KB+ |
| 3D physics simulation | @react-three/rapier | +50KB |
| 2D physics (gravity, collisions) | Matter.js | ~75KB |
| Post-processing effects | postprocessing (R3F) | varies |

### Decision Guide
- If CSS can do it (gradients, simple animations) → **don't use WebGL**
- If you need one shader background → **ogl** (lightest)
- If you need multiple 3D objects or physics → **Three.js + R3F**
- If you need 2D physics (falling text, bouncing) → **Matter.js**

## React Three Fiber (R3F) Pattern

```tsx
import { Canvas } from "@react-three/fiber"
import { Environment, Float, OrbitControls } from "@react-three/drei"

function Scene() {
  return (
    <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
      <Environment preset="city" />
      <OrbitControls enableZoom={false} />
      <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
        <mesh>
          <sphereGeometry args={[1, 64, 64]} />
          <meshStandardMaterial color="#22c55e" metalness={0.8} roughness={0.2} />
        </mesh>
      </Float>
    </Canvas>
  )
}
```

### With Physics (Rapier)
```tsx
import { Physics, RigidBody } from "@react-three/rapier"

<Canvas>
  <Physics gravity={[0, -9.81, 0]}>
    <RigidBody type="dynamic">
      <mesh><sphereGeometry /><meshStandardMaterial /></mesh>
    </RigidBody>
    <RigidBody type="fixed">
      <mesh><planeGeometry args={[10, 10]} /><meshStandardMaterial /></mesh>
    </RigidBody>
  </Physics>
</Canvas>
```

## ogl Shader Pattern

Lightweight alternative to Three.js for 2D shader effects:

```tsx
import { Renderer, Camera, Program, Mesh, Triangle } from "ogl"

useEffect(() => {
  const renderer = new Renderer({ canvas: canvasRef.current, alpha: true })
  const gl = renderer.gl
  const camera = new Camera(gl)

  const geometry = new Triangle(gl)
  const program = new Program(gl, {
    vertex: `...`,    // GLSL vertex shader
    fragment: `...`,  // GLSL fragment shader
    uniforms: {
      uTime: { value: 0 },
      uResolution: { value: [gl.canvas.width, gl.canvas.height] }
    }
  })

  const mesh = new Mesh(gl, { geometry, program })

  function animate(t) {
    program.uniforms.uTime.value = t * 0.001
    renderer.render({ scene: mesh, camera })
    requestAnimationFrame(animate)
  }
  requestAnimationFrame(animate)
}, [])
```

## Canvas API Pattern (Simple Particles)

```tsx
useEffect(() => {
  const canvas = canvasRef.current
  const ctx = canvas.getContext("2d")
  const particles = Array.from({ length: 100 }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - 0.5) * 0.5,
    vy: (Math.random() - 0.5) * 0.5,
    size: Math.random() * 2 + 1
  }))

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    particles.forEach(p => {
      p.x += p.vx
      p.y += p.vy
      // Wrap around edges
      if (p.x < 0) p.x = canvas.width
      if (p.x > canvas.width) p.x = 0
      if (p.y < 0) p.y = canvas.height
      if (p.y > canvas.height) p.y = 0

      ctx.fillStyle = "rgba(255, 255, 255, 0.5)"
      ctx.beginPath()
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
      ctx.fill()
    })
    requestAnimationFrame(animate)
  }
  animate()
}, [])
```

## Performance Guidelines

### General
- Limit to 1 shader/WebGL canvas per viewport
- Target 60fps; if dropping below 30fps, simplify or disable
- Use `Intersection Observer` to pause off-screen canvases
- Reduce canvas resolution on mobile: `renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))`

### Three.js Specific
- Use `instancedMesh` for repeated objects (huge performance gain)
- Dispose geometries, materials, and textures when unmounting
- Use `Suspense` for lazy loading 3D assets
- Prefer `drei` helpers over manual setup

### Mobile Considerations
- Always provide a static fallback for devices without WebGL
- Reduce particle count by 50-75% on mobile
- Disable post-processing effects on mobile
- Consider using a static image or CSS animation instead
- Test on actual mobile devices (not just browser responsive mode)

### Memory Management
```tsx
useEffect(() => {
  return () => {
    // Clean up on unmount
    geometry.dispose()
    material.dispose()
    texture.dispose()
    renderer.dispose()
  }
}, [])
```

## Anti-Patterns
- DON'T: Use Three.js for simple gradient animations (use CSS)
- DON'T: Load 3D assets without lazy loading / Suspense
- DON'T: Skip mobile fallback for WebGL-heavy effects
- DON'T: Keep rendering off-screen canvases
- DON'T: Use full devicePixelRatio on high-DPI mobile (3x = very expensive)
- DO: Measure GPU memory and provide quality tiers
- DO: Use `will-change: transform` sparingly and remove after animation
- DO: Provide a "reduce effects" toggle for performance-sensitive users
