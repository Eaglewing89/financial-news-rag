
## Testing

### Agent assistance warning
Note: Be VERY vigilant if you let Copilot or similar agents review your tests and/or create them. I have found several instances of the following behaviour despite clear instruction to not do this: 

#### In general:
- Additions to source code to pass hallucinated tests

#### Unit test:
- Breaking encapsulation by importing other classes
- Making actual external API calls without proper mocking

### Testing Checklists

#### **Quality Standard Checklist for All Unit Tests:**

When reviewing or creating unit tests, they must meet these criteria:

**Structure:**
- [ ] Clear module docstring explaining purpose and scope
- [ ] Logical class-based organization by functional area
- [ ] Each test class has descriptive docstring
- [ ] Clean, minimal imports

**Isolation & Dependencies:**
- [ ] Tests only the target component (true unit testing)
- [ ] All external dependencies properly mocked
- [ ] No actual API calls to external services
- [ ] Uses in-memory alternatives where possible

**Fixtures & Data:**
- [ ] Uses shared fixtures from conftest.py appropriately
- [ ] Creates component-specific fixtures within test classes
- [ ] Uses test data factories instead of hardcoded data
- [ ] Ensures test isolation (no shared state between tests)

**Test Quality:**
- [ ] Descriptive test method names
- [ ] Clear test method docstrings
- [ ] Follows AAA pattern (Arrange, Act, Assert)
- [ ] Tests one behavior per method
- [ ] Comprehensive assertions
- [ ] Includes edge cases and error scenarios

**pytest Standards:**
- [ ] Pure pytest style (no unittest.TestCase)
- [ ] Proper fixture dependency injection
- [ ] Uses pytest.raises() for exception testing
- [ ] Simple assert statements

**Performance:**
- [ ] Fast execution (< 1 second per test typically)
- [ ] Minimal setup overhead
- [ ] No slow external operations
