# Common Bug Patterns & Solutions

Quick reference guide for frequently encountered bugs and their fixes.

---

## Frontend Bugs

### 1. Undefined/Null Errors

**Problem**: `Cannot read property 'X' of undefined`

```typescript
// Crash when data not loaded yet
const name = user.profile.name;
```

**Solutions**:

```typescript
// Option 1: Optional chaining + nullish coalescing
const name = user?.profile?.name ?? 'Unknown';

// Option 2: Conditional rendering
{user?.profile && <div>{user.profile.name}</div>}

// Option 3: Early return
if (!user?.profile) return <div>Loading...</div>;
```

---

### 2. Stale Closures in useEffect

**Problem**: Event handlers/callbacks use old state values

```typescript
function Counter() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      console.log(count); // Always logs 0!
    }, 1000);
    return () => clearInterval(interval);
  }, []); // Missing dependency

  return <button onClick={() => setCount(c => c + 1)}>+</button>;
}
```

**Solutions**:

```typescript
// Option 1: Include dependency
useEffect(() => {
  const interval = setInterval(() => {
    console.log(count); // Now updates!
  }, 1000);
  return () => clearInterval(interval);
}, [count]); // Dependency added

// Option 2: Use functional update
useEffect(() => {
  const interval = setInterval(() => {
    setCount(c => {
      console.log(c); // Current value
      return c;
    });
  }, 1000);
  return () => clearInterval(interval);
}, []); // Can stay empty

// Option 3: Use ref for latest value
const countRef = useRef(count);
countRef.current = count;

useEffect(() => {
  const interval = setInterval(() => {
    console.log(countRef.current); // Latest value
  }, 1000);
  return () => clearInterval(interval);
}, []);
```

---

### 3. Missing Cleanup in useEffect

**Problem**: Memory leaks from subscriptions/listeners

```typescript
useEffect(() => {
  const subscription = api.subscribe(data => setData(data));
  // Missing cleanup!
}, []);
```

**Solution**:

```typescript
useEffect(() => {
  const subscription = api.subscribe(data => setData(data));

  return () => {
    subscription.unsubscribe(); // Cleanup
  };
}, []);
```

**Common things that need cleanup**:
- Event listeners (`addEventListener`)
- Intervals (`setInterval`)
- Timeouts (`setTimeout`)
- Subscriptions (WebSockets, observables)
- API requests (cancellation tokens)

---

### 4. Race Conditions in Async Effects

**Problem**: Old requests overwrite new ones

```typescript
useEffect(() => {
  fetchUser(userId).then(setUser);
  // If userId changes quickly, old responses arrive after new ones
}, [userId]);
```

**Solution**:

```typescript
useEffect(() => {
  let cancelled = false;

  fetchUser(userId).then(user => {
    if (!cancelled) {
      setUser(user);
    }
  });

  return () => {
    cancelled = true;
  };
}, [userId]);
```

---

### 5. Infinite Re-render Loops

**Problem**: Component re-renders infinitely

```typescript
function Component() {
  const [data, setData] = useState([]);

  useEffect(() => {
    setData([...data, 'new']); // Triggers effect again!
  }, [data]); // Dependency causes loop

  return <div>{data.length}</div>;
}
```

**Solutions**:

```typescript
// Option 1: Remove problematic dependency
useEffect(() => {
  setData(prevData => [...prevData, 'new']);
}, []); // Empty deps - runs once

// Option 2: Use ref instead of state
const dataRef = useRef([]);

useEffect(() => {
  dataRef.current = [...dataRef.current, 'new'];
}, []);

// Option 3: Add condition
useEffect(() => {
  if (data.length === 0) { // Only run when empty
    setData(['new']);
  }
}, [data]);
```

---

### 6. Key Prop Issues in Lists

**Problem**: List items reordering incorrectly

```typescript
// Using index as key
{todos.map((todo, index) => (
  <TodoItem key={index} todo={todo} />
))}
```

**Solution**:

```typescript
// Use stable, unique ID
{todos.map(todo => (
  <TodoItem key={todo.id} todo={todo} />
))}

// If no ID, generate stable key
{todos.map((todo, index) => (
  <TodoItem key={`${todo.title}-${index}`} todo={todo} />
))}
```

---

### 7. Form Input Controlled/Uncontrolled Switch

**Problem**: `Warning: A component is changing an uncontrolled input to be controlled`

```typescript
const [value, setValue] = useState(); // undefined initially

<input value={value} onChange={e => setValue(e.target.value)} />
```

**Solution**:

```typescript
// Initialize with empty string
const [value, setValue] = useState('');

<input value={value} onChange={e => setValue(e.target.value)} />

// Or use defaultValue for uncontrolled
<input defaultValue="" onChange={e => console.log(e.target.value)} />
```

---

## Backend Bugs

### 1. SQL Injection

**Problem**: User input directly in SQL query

```python
# DANGEROUS!
email = request.args.get('email')
query = f"SELECT * FROM users WHERE email = '{email}'"
db.execute(query)
# User can input: ' OR '1'='1
```

**Solution**:

```python
# Use parameterized queries
from sqlalchemy import text

email = request.args.get('email')
query = text("SELECT * FROM users WHERE email = :email")
result = db.execute(query, {"email": email})

# Or use ORM
user = db.query(User).filter(User.email == email).first()
```

---

### 2. N+1 Query Problem

**Problem**: One query per item in a loop

```python
# 1 query to get todos
todos = db.query(Todo).all()

# N queries (one per todo)
for todo in todos:
    user = db.query(User).filter(User.id == todo.user_id).first()
    print(f"{todo.title} by {user.name}")
```

**Solution**:

```python
# Use JOIN - single query
from sqlalchemy.orm import joinedload

todos = db.query(Todo).options(joinedload(Todo.user)).all()

for todo in todos:
    print(f"{todo.title} by {todo.user.name}") # No extra query
```

---

### 3. Missing Authentication Check

**Problem**: Protected endpoint accessible without auth

```python
@app.get("/api/admin/users")
async def get_all_users(db: DatabaseDep):
    return db.query(User).all() # Anyone can access!
```

**Solution**:

```python
@app.get("/api/admin/users")
async def get_all_users(
    db: DatabaseDep,
    current_user: User = Depends(get_current_user) # Require auth
):
    if current_user.role != "admin": # Check role
        raise HTTPException(403, "Admin only")
    return db.query(User).all()
```

---

### 4. Missing Input Validation

**Problem**: Invalid data causes errors

```python
@app.post("/api/users")
async def create_user(email: str, age: int):
    # No validation!
    user = User(email=email, age=age)
    db.add(user)
    db.commit()
```

**Solution**:

```python
from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    email: EmailStr # Validates email format
    age: int = Field(ge=0, le=150) # Must be 0-150

@app.post("/api/users")
async def create_user(user: UserCreate):
    # Pydantic validates automatically
    db_user = User(**user.model_dump())
    db.add(db_user)
    db.commit()
```

---

### 5. Unhandled Exceptions

**Problem**: Server crashes on error

```python
@app.post("/api/todos")
async def create_todo(todo: TodoCreate, user: User = Depends(get_current_user)):
    db_todo = Todo(**todo.model_dump(), user_id=user.id)
    db.add(db_todo)
    db.commit() # Could fail!
    return db_todo
```

**Solution**:

```python
from fastapi import HTTPException

@app.post("/api/todos")
async def create_todo(todo: TodoCreate, user: User = Depends(get_current_user)):
    try:
        db_todo = Todo(**todo.model_dump(), user_id=user.id)
        db.add(db_todo)
        db.commit()
        db.refresh(db_todo)
        return db_todo
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(409, "Duplicate entry")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create todo: {e}")
        raise HTTPException(500, "Internal server error")
```

---

### 6. Missing CORS Configuration

**Problem**: Frontend can't call API

```
Access to fetch at 'http://localhost:8000/api/todos' from origin
'http://localhost:3000' has been blocked by CORS policy
```

**Solution**:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For production, be specific:
# allow_origins=["https://yourdomain.com"]
```

---

### 7. Password Storage

**Problem**: Passwords stored in plain text

```python
user = User(email=email, password=password) # NEVER DO THIS!
```

**Solution**:

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hash before storing
password_hash = pwd_context.hash(password)
user = User(email=email, password_hash=password_hash)

# Verify on login
if not pwd_context.verify(plain_password, user.password_hash):
    raise HTTPException(401, "Invalid credentials")
```

---

## Mobile Bugs

### 1. Memory Leaks in Flutter

**Problem**: Controllers not disposed

```dart
class MyWidget extends StatefulWidget {
  @override
  _MyWidgetState createState() => _MyWidgetState();
}

class _MyWidgetState extends State<MyWidget> {
  final controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return TextField(controller: controller);
  }
  // Missing dispose!
}
```

**Solution**:

```dart
class _MyWidgetState extends State<MyWidget> {
  final controller = TextEditingController();

  @override
  void dispose() {
    controller.dispose(); // Clean up
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return TextField(controller: controller);
  }
}
```

---

### 2. Platform-Specific Code Not Checked

**Problem**: iOS-specific code crashes on Android

```dart
// Crashes on Android
import 'dart:io' show Platform;

final deviceName = Platform.isIOS ? 'iPhone' : 'Unknown';
```

**Solution**:

```dart
import 'dart:io' show Platform;

final deviceName = Platform.isIOS
    ? 'iPhone'
    : Platform.isAndroid
        ? 'Android'
        : 'Unknown';

// Or use conditional imports
if (Platform.isIOS) {
  // iOS-specific code
} else if (Platform.isAndroid) {
  // Android-specific code
}
```

---

## Performance Bugs

### 1. Unnecessary Re-renders (React)

**Problem**: Component re-renders on every parent render

```typescript
function Parent() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <button onClick={() => setCount(c => c + 1)}>Count: {count}</button>
      <ExpensiveChild data={someData} /> {/* Re-renders every time! */}
    </div>
  );
}
```

**Solution**:

```typescript
// Memoize the expensive component
const ExpensiveChild = React.memo(function ExpensiveChild({ data }) {
  // Only re-renders when data changes
  return <div>{/* expensive computation */}</div>;
});

// Or memoize the props
function Parent() {
  const [count, setCount] = useState(0);
  const memoizedData = useMemo(() => computeData(), []);

  return (
    <div>
      <button onClick={() => setCount(c => c + 1)}>Count: {count}</button>
      <ExpensiveChild data={memoizedData} />
    </div>
  );
}
```

---

### 2. Large Bundle Size

**Problem**: Importing entire library

```typescript
// Imports all of lodash (~70KB)
import _ from 'lodash';

const unique = _.uniq(array);
```

**Solution**:

```typescript
// Import only what you need
import uniq from 'lodash/uniq'; // ~2KB

const unique = uniq(array);

// Or use native methods
const unique = [...new Set(array)];
```

---

## Security Bugs

### 1. XSS (Cross-Site Scripting)

**Problem**: User input rendered as HTML

```typescript
// Dangerous!
<div dangerouslySetInnerHTML={{ __html: userComment }} />
```

**Solution**:

```typescript
// React escapes by default
<div>{userComment}</div>

// If HTML needed, sanitize first
import DOMPurify from 'dompurify';

<div dangerouslySetInnerHTML={{
  __html: DOMPurify.sanitize(userComment)
}} />
```

---

### 2. Missing Rate Limiting

**Problem**: API can be abused

```python
@app.post("/api/auth/login")
async def login(credentials: LoginRequest):
    # No rate limiting - brute force possible!
    ...
```

**Solution**:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/auth/login")
@limiter.limit("5/minute") # Max 5 attempts per minute
async def login(request: Request, credentials: LoginRequest):
    ...
```

---

## Common Error Messages & Solutions

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `Cannot read property 'X' of undefined` | Accessing property before data loads | Add null check or optional chaining |
| `Maximum update depth exceeded` | Infinite re-render loop | Check useEffect dependencies |
| `Warning: Each child should have unique key` | Missing/duplicate keys in list | Use unique, stable IDs as keys |
| `401 Unauthorized` | Missing/invalid auth token | Check token in request headers |
| `403 Forbidden` | Insufficient permissions | Verify user role/permissions |
| `404 Not Found` | Wrong URL or resource doesn't exist | Check endpoint path and resource ID |
| `500 Internal Server Error` | Backend exception | Check server logs for stack trace |
| `CORS Error` | Cross-origin request blocked | Configure CORS middleware |
| `ERR_CONNECTION_REFUSED` | Server not running | Start the backend server |

---

## Quick Debugging Commands

### Frontend
```bash
# Check for errors
npm run lint

# Run tests
npm test

# Build and check bundle size
npm run build
npm run analyze

# Check for unused dependencies
npx depcheck
```

### Backend
```bash
# Check for security issues
pip install safety
safety check

# Run tests with coverage
pytest --cov=app

# Check for SQL injection
bandit -r app/
```

### Mobile
```bash
# Check for issues
flutter analyze

# Run tests
flutter test

# Check app size
flutter build apk --analyze-size
```

---

## When to Use Each Agent

| Bug Type | Best Agent | Reason |
|----------|-----------|---------|
| Frontend crash | debug-agent | Specializes in bug diagnosis |
| Backend API error | debug-agent | Can trace through stack |
| Complex multi-domain | oma-coordination | Coordinates multiple agents |
| Security vulnerability | qa-agent | Security expertise |
| Performance issue | qa-agent | Performance profiling tools |
| New feature needed | Specialist agent | Not a bug, it's a feature |

---

## Prevention Tips

1. **Write tests first** - Catch bugs before they ship
2. **Use TypeScript** - Catch type errors at compile time
3. **Enable strict mode** - More safety checks
4. **Code review** - Second pair of eyes
5. **Automated linting** - Enforce best practices
6. **Error monitoring** - Sentry, LogRocket, etc.
7. **User testing** - Real users find real bugs

---

**Remember**: The best bug is the one that never happens. Write defensive code, test thoroughly, and document lessons learned!
