"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class TransactionCreate(BaseModel):
    user_id: str = Field(..., example="user_001")
    merchant_id: str = Field(..., example="merchant_001")
    amount: Decimal = Field(..., gt=0, example=99.99)
    currency: str = Field(default="EUR", example="EUR")
    payment_method: str = Field(..., example="card")
    card_last4: Optional[str] = Field(None, example="1234")
    ip_address: Optional[str] = Field(None, example="82.65.123.45")
    device_type: Optional[str] = Field(default="mobile", example="mobile")
    
    @validator('currency')
    def validate_currency(cls, v):
        allowed = ['EUR', 'USD', 'GBP', 'CAD', 'AUD']
        if v not in allowed:
            raise ValueError(f'Currency must be one of {allowed}')
        return v

class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: Decimal
    currency: str
    status: str
    payment_method: str
    card_last4: Optional[str]
    is_fraud: bool
    fraud_score: int
    ml_fraud_probability: Optional[float]
    created_at: datetime
    ip_address: Optional[str]
    device_type: Optional[str]

class FraudScore(BaseModel):
    transaction_id: str
    fraud_score: int
    probability: float
    is_fraud: bool
    rules_triggered: List[str] = []

class User(BaseModel):
    user_id: str
    email: str
    name: str
    country: str
    total_transactions: int
    avg_amount: Decimal
    is_active: bool

class Merchant(BaseModel):
    merchant_id: str
    name: str
    category: str
    country: str
    risk_score: int
    is_active: bool
