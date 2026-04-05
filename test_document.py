"""
Test Document Generator

Creates sample documents for testing the RAG service.
Generates documents with known content that's easy to query.
"""


def create_sample_txt():
    """
    Create a sample TXT file about a fictional company policy.
    
    This creates a document that's easy to test with specific questions.
    """
    content = """
ACME Corporation Employee Handbook
Last Updated: December 2024

SECTION 1: REFUND POLICY

Our refund policy is designed to ensure customer satisfaction. All purchases 
can be refunded within 30 days of the original purchase date. To request a 
refund, customers must provide their original receipt and the product must 
be in its original condition.

For digital products, refunds are available within 14 days. The refund will 
be processed within 5-7 business days after approval.

SECTION 2: VACATION POLICY

All full-time employees are entitled to 15 days of paid vacation per year. 
Vacation days accrue at a rate of 1.25 days per month. Employees must submit 
vacation requests at least 2 weeks in advance through the HR portal.

Unused vacation days can be carried over to the next year, up to a maximum 
of 5 days. Vacation days cannot be exchanged for cash.

SECTION 3: REMOTE WORK POLICY

Employees may work remotely up to 3 days per week, subject to manager approval. 
Remote work arrangements must be documented in writing and reviewed quarterly.

All remote workers must:
- Have a dedicated workspace
- Maintain regular business hours
- Be available for video calls
- Attend in-person meetings when required

Equipment for remote work, including laptops and monitors, will be provided 
by the company. Employees are responsible for maintaining a secure work 
environment.

SECTION 4: BENEFITS

Health Insurance:
- Company covers 80% of premiums
- Coverage starts on the first day of employment
- Includes medical, dental, and vision
- Family coverage available

Retirement:
- 401(k) plan with 4% company match
- Immediate vesting
- Financial planning resources available

Professional Development:
- $2,000 annual learning budget
- Conference attendance encouraged
- Internal training programs
- Tuition reimbursement up to $5,000 per year

SECTION 5: CODE OF CONDUCT

All employees are expected to:
- Treat colleagues with respect
- Maintain confidentiality
- Follow security protocols
- Report violations promptly

Violations of the code of conduct may result in disciplinary action, 
including termination of employment.

SECTION 6: CONTACT INFORMATION

Human Resources: hr@acme.com
IT Support: support@acme.com
Emergency: 911

For questions about this handbook, please contact HR at hr@acme.com or 
call (555) 123-4567.
"""
    
    with open("test_employee_handbook.txt", "w") as f:
        f.write(content)
    
    print("✅ Created: test_employee_handbook.txt")
    print("   This document contains information about:")
    print("   - Refund policy")
    print("   - Vacation policy")
    print("   - Remote work policy")
    print("   - Benefits")
    print("   - Code of conduct")


def create_sample_technical_doc():
    """
    Create a sample technical document about a fictional API.
    """
    content = """
TechAPI Documentation
Version 2.0

INTRODUCTION

TechAPI is a RESTful API service for managing user data and authentication.
This documentation covers all endpoints, authentication methods, and best practices.

AUTHENTICATION

All API requests require authentication using an API key. Include your API key 
in the request header:

Authorization: Bearer YOUR_API_KEY

API keys can be generated from your dashboard at https://dashboard.techapi.com

Rate Limiting:
- Free tier: 100 requests per hour
- Pro tier: 1,000 requests per hour
- Enterprise: Unlimited requests

ENDPOINTS

1. User Management

GET /api/v2/users
List all users in your organization.

Query Parameters:
- page: Page number (default: 1)
- limit: Results per page (default: 10, max: 100)
- sort: Sort field (created_at, name, email)

Response: 200 OK
{
  "users": [...],
  "total": 150,
  "page": 1
}

POST /api/v2/users
Create a new user.

Request Body:
{
  "email": "user@example.com",
  "name": "John Doe",
  "role": "member"
}

Response: 201 Created
{
  "id": "usr_123",
  "email": "user@example.com",
  "created_at": "2024-12-05T10:00:00Z"
}

2. Authentication

POST /api/v2/auth/login
Authenticate a user and receive an access token.

Request Body:
{
  "email": "user@example.com",
  "password": "secure_password"
}

Response: 200 OK
{
  "access_token": "eyJ0eXAi...",
  "expires_in": 3600
}

ERROR CODES

400 Bad Request - Invalid request parameters
401 Unauthorized - Missing or invalid API key
403 Forbidden - Insufficient permissions
404 Not Found - Resource not found
429 Too Many Requests - Rate limit exceeded
500 Internal Server Error - Server error

BEST PRACTICES

1. Always use HTTPS for API requests
2. Store API keys securely (use environment variables)
3. Implement exponential backoff for rate limiting
4. Cache responses when appropriate
5. Use pagination for large datasets

SUPPORT

For technical support:
- Email: support@techapi.com
- Discord: discord.gg/techapi
- Documentation: docs.techapi.com

For billing inquiries:
- Email: billing@techapi.com
- Phone: 1-800-TECH-API
"""
    
    with open("test_api_documentation.txt", "w") as f:
        f.write(content)
    
    print("✅ Created: test_api_documentation.txt")
    print("   This document contains technical API information")


def create_sample_faq():
    """
    Create a sample FAQ document.
    """
    content = """
Frequently Asked Questions
CloudHost Streaming Service

GENERAL QUESTIONS

Q: What is CloudHost?
A: CloudHost is a premium streaming service offering unlimited access to 
thousands of movies, TV shows, and original content. Stream on any device, 
anywhere, anytime.

Q: How much does CloudHost cost?
A: We offer three plans:
   - Basic: $9.99/month (1 screen, SD quality)
   - Standard: $14.99/month (2 screens, HD quality)
   - Premium: $19.99/month (4 screens, 4K quality)

Q: Can I cancel anytime?
A: Yes! There are no long-term contracts or cancellation fees. You can cancel 
your subscription at any time through your account settings.

TECHNICAL QUESTIONS

Q: What devices are supported?
A: CloudHost works on:
   - Smart TVs (Samsung, LG, Sony)
   - Streaming devices (Roku, Fire TV, Apple TV)
   - Mobile devices (iOS 12+, Android 8+)
   - Web browsers (Chrome, Firefox, Safari, Edge)
   - Game consoles (PlayStation, Xbox)

Q: What internet speed do I need?
A: Recommended speeds:
   - SD quality: 3 Mbps
   - HD quality: 5 Mbps
   - 4K quality: 25 Mbps

Q: Can I download content for offline viewing?
A: Yes, Standard and Premium subscribers can download selected titles on mobile 
devices. Downloads are available for 30 days, and you have 48 hours to finish 
watching once you start.

ACCOUNT QUESTIONS

Q: How many profiles can I create?
A: You can create up to 5 profiles per account. Each profile has its own 
watchlist and recommendations.

Q: Can I share my account?
A: Your account can be used by people in your household. The number of 
simultaneous streams depends on your plan.

Q: How do I reset my password?
A: Click "Forgot Password" on the login page. We'll send a reset link to 
your registered email address.

BILLING QUESTIONS

Q: When will I be charged?
A: Your subscription renews automatically on the same day each month. You'll 
receive an email reminder 3 days before billing.

Q: What payment methods do you accept?
A: We accept:
   - Credit cards (Visa, Mastercard, Amex)
   - Debit cards
   - PayPal
   - Gift cards

Q: Can I get a refund?
A: If you cancel within the first 7 days of your initial subscription, you 
can request a full refund. Contact support@cloudhost.com

CONTENT QUESTIONS

Q: How often is new content added?
A: We add new movies and shows every week. Check the "New Releases" section 
for the latest additions.

Q: Are there ads?
A: No, CloudHost is completely ad-free on all plans.

Q: Can I request content?
A: Yes! Use the feedback form in your account to suggest titles you'd like 
to see added.

Still have questions? Contact our 24/7 support team at support@cloudhost.com 
or call 1-888-CLOUD-01
"""
    
    with open("test_faq_streaming.txt", "w") as f:
        f.write(content)
    
    print("✅ Created: test_faq_streaming.txt")
    print("   This document contains FAQ about a streaming service")


def print_test_questions():
    """
    Print suggested test questions for each document.
    """
    print("\n" + "=" * 60)
    print("SUGGESTED TEST QUESTIONS")
    print("=" * 60)
    
    print("\nFor test_employee_handbook.txt:")
    print("  - What is the refund policy?")
    print("  - How many vacation days do employees get?")
    print("  - What is the remote work policy?")
    print("  - What benefits are included?")
    print("  - How much is the learning budget?")
    
    print("\nFor test_api_documentation.txt:")
    print("  - How do I authenticate with the API?")
    print("  - What are the rate limits?")
    print("  - How do I create a new user?")
    print("  - What error codes might I receive?")
    print("  - What is the endpoint for user login?")
    
    print("\nFor test_faq_streaming.txt:")
    print("  - How much does CloudHost cost?")
    print("  - What devices are supported?")
    print("  - Can I download content for offline viewing?")
    print("  - What payment methods are accepted?")
    print("  - Is there a refund policy?")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n📝 Creating test documents...\n")
    
    create_sample_txt()
    create_sample_technical_doc()
    create_sample_faq()
    
    print_test_questions()
    
    print("\n✅ All test documents created!")
    print("\nNext steps:")
    print("1. Start the RAG service: python main.py")
    print("2. Upload a test document using test_client.py")
    print("3. Try the suggested questions above")