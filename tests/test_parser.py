"""Tests for the parser module."""

import pytest
from cards_to_contacts.parser import parse_contact


def test_basic_business_card():
    """Test parsing a basic business card layout."""
    raw_text = """John Smith
Senior Software Engineer
Tech Solutions Inc.
john.smith@techsolutions.com
(555) 123-4567
www.techsolutions.com
123 Main St
Anytown, NY 12345"""
    
    contact = parse_contact(raw_text)
    
    assert contact.full_name == "John Smith"
    assert contact.title == "Senior Software Engineer"
    assert contact.company == "Tech Solutions Inc."
    assert "john.smith@techsolutions.com" in contact.emails
    assert len(contact.phones) == 1
    assert contact.website == "www.techsolutions.com"
    assert "123 Main St" in contact.address


def test_mixed_order_business_card():
    """Test parsing when information is not in standard order."""
    raw_text = """Tech Solutions Inc.
John Smith
Senior Software Engineer
john.smith@techsolutions.com
(555) 123-4567"""
    
    contact = parse_contact(raw_text)
    
    assert contact.full_name == "John Smith"
    assert contact.title == "Senior Software Engineer"
    assert contact.company == "Tech Solutions Inc."


def test_title_recognition():
    """Test recognition of various job titles."""
    test_cases = [
        ("CEO", "Chief Executive Officer"),
        ("VP Marketing", "Vice President of Marketing"),
        ("Software Developer", "Lead Developer"),
        ("Sales Manager", "Regional Sales Manager"),
    ]
    
    for title_text, expected_title in test_cases:
        raw_text = f"""Jane Doe
{title_text}
ACME Corp
jane@acme.com"""
        
        contact = parse_contact(raw_text)
        assert contact.title == title_text


def test_company_recognition():
    """Test recognition of company names with various suffixes."""
    test_cases = [
        "ACME Corporation",
        "Tech Solutions LLC",
        "Global Enterprises Inc.",
        "Consulting Partners Ltd",
    ]
    
    for company_name in test_cases:
        raw_text = f"""John Doe
Manager
{company_name}
john@company.com"""
        
        contact = parse_contact(raw_text)
        assert contact.company == company_name


def test_phone_number_variations():
    """Test parsing of various phone number formats."""
    phone_formats = [
        "(555) 123-4567",
        "555-123-4567",
        "555.123.4567",
        "5551234567",
        "+1 555 123 4567",
    ]
    
    for phone in phone_formats:
        raw_text = f"""John Smith
Manager
ACME Corp
{phone}
john@acme.com"""
        
        contact = parse_contact(raw_text)
        assert len(contact.phones) == 1


def test_minimal_business_card():
    """Test parsing a minimal business card with just name and contact."""
    raw_text = """John Smith
john@email.com
555-123-4567"""
    
    contact = parse_contact(raw_text)
    
    assert contact.full_name == "John Smith"
    assert "john@email.com" in contact.emails
    assert len(contact.phones) == 1


def test_empty_text():
    """Test parsing empty or whitespace-only text."""
    contact = parse_contact("")
    assert contact.full_name is None
    assert len(contact.emails) == 0
    assert len(contact.phones) == 0
    
    contact = parse_contact("   \n\n  \t  ")
    assert contact.full_name is None