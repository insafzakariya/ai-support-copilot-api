# Acme Cloud Support Handbook

**Document owner:** Support Operations  
**Version:** 2.0  
**Last updated:** 2026-11-12

This handbook contains the approved support guidance for Acme Cloud customer tickets. Use it to guide troubleshooting and customer communication. If a situation is not covered here, tell the customer that a human specialist must investigate.

## 1. Payment succeeded but the subscription is inactive

When a customer says that payment succeeded but their subscription is still inactive:

1. Verify the payment provider status using the transaction ID.
2. Confirm whether the payment is captured or still pending.
3. Retry subscription provisioning if the payment is captured.
4. Confirm the subscription status after provisioning is retried.

Do not ask the customer to pay again while the original payment is being investigated. Do not promise an exact activation time unless the provider has supplied one.

## 2. Duplicate charges

When a customer reports duplicate charges:

1. Ask for the transaction IDs or the last four digits of the payment method, never the full card number.
2. Check whether one of the charges is still pending.
3. If only one charge completed, explain that the pending authorization may disappear automatically.
4. If two charges completed, create a refund request for the duplicate charge.
5. Tell the customer that refund processing time depends on the payment provider. Do not promise a guaranteed date.

## 3. Failed payments

For a failed payment:

- Confirm the failure status and provider error category.
- Ask the customer to check the card expiry date and billing address in the account settings.
- Suggest trying another approved payment method if appropriate.
- Never ask for the full card number, CVV, password, or one-time password.
- Do not retry a payment repeatedly because repeated retries may create additional authorizations.

If the customer believes they were charged despite a failed status, escalate the case with the transaction ID and provider response code.

## 4. Refund policy

Support agents may submit refund requests for confirmed duplicate completed charges. A refund request is not a guarantee that the refund will be approved.

For any other refund request:

1. Review the account and transaction details.
2. Check whether the request meets the applicable account or plan policy.
3. Escalate to a human billing specialist when policy eligibility is unclear.

Never invent a refund amount, approval, or processing timeline.

## 5. Account access and password issues

For account access problems:

- Direct the customer to the official password-reset flow.
- Do not request or accept passwords, recovery codes, one-time passwords, or authentication secrets.
- Do not disable security controls based only on a customer message.
- Escalate suspected account takeover, repeated failed login attempts, or changes to account ownership.

## 6. Service incidents and outages

When a customer reports an outage or repeated errors:

1. Check the official service-status information.
2. Record the affected feature, approximate start time, timezone, and error message.
3. Do not claim that an outage exists unless it is confirmed by an approved status source.
4. If no incident is confirmed, provide basic troubleshooting and escalate recurring problems.

Do not promise a resolution time unless an approved incident update provides one.

## 7. Product features and information

When a customer asks about Acme Cloud features:

- Direct them to the official product documentation at `docs.acmecloud.com`
- Highlight the core features: multi-cloud orchestration, automated scaling, real-time monitoring, and API-first architecture
- For feature comparisons or detailed technical specs, provide the link to the feature matrix
- If a feature is not documented, do not invent capabilities; escalate to the product team for clarification
- For questions about roadmap or upcoming features, direct them to the public roadmap or escalate to product management

## 8. Payment methods and accepted payment types

Acme Cloud accepts the following payment methods:

- Credit cards (Visa, Mastercard, American Express, Discover)
- Bank transfers (ACH for US customers, international wire transfers)
- Purchase orders (for enterprise accounts with approved credit)

For customers asking about payment options:

1. Direct them to the account billing settings where they can view and update payment methods
2. For new payment method setup, provide the link to the secure payment portal
3. If a customer's preferred payment method is not supported, escalate to billing for possible exceptions
4. Never request full payment card information; use the secure portal instead

## 9. Warranty period and coverage

Acme Cloud offers the following warranty terms:

- **Service Level Agreement (SLA)**: 99.9% uptime for production deployments on the Pro plan and above
- **Hardware Warranty**: 12 months from account activation for any infrastructure or appliance deployments
- **Support Warranty**: Full technical support access for the duration of the active subscription; support ends 30 days after subscription cancellation
- **Data Warranty**: Customer data is maintained with redundancy; backups are retained for 30 days after deletion per the Data Retention Policy

When a customer asks about warranty:

1. Clarify which warranty type they're asking about (SLA, hardware, support, or data)
2. Direct them to the Terms of Service for comprehensive warranty details
3. For warranty claims, escalate to the warranty team with details about the incident and timeline
4. Do not promise coverage or exceptions; warranty eligibility must be verified

## 10. Contacting customer support

Support contact information:

- **Email**: support@acmecloud.com (response time: 4 hours for priority, 24 hours for standard)
- **Phone**: 1-800-ACME-HELP (Monday–Friday, 8am–6pm PT)
- **Chat**: Available in the customer portal during business hours
- **Status Page**: status.acmecloud.com for real-time incident updates
- **Community Forum**: community.acmecloud.com for peer support and discussions

For customers contacting support:

1. Direct them to the appropriate channel based on urgency (chat/phone for critical issues, email for documentation requests)
2. If they reach you through an unsupported channel, provide the official support contact information
3. Escalate security incidents (suspected breach, unauthorized access) directly to security@acmecloud.com
4. For billing disputes, escalate to billing@acmecloud.com

## 11. System requirements and installation

Acme Cloud system requirements:

- **Minimum**: 2 vCPU, 4GB RAM, 20GB storage for development environments
- **Recommended**: 4+ vCPU, 8GB+ RAM, 100GB+ storage for production deployments
- **Supported OS**: Linux (Ubuntu 20.04+, CentOS 8+), macOS 10.15+, Windows Server 2019+
- **Network**: Minimum 1Mbps internet connection; 10Mbps+ recommended for production
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+ for the web console

For installation guidance:

1. Direct customers to the official installation guide at `docs.acmecloud.com/install`
2. If system requirements are not met, provide a clear explanation of what needs to be upgraded
3. For custom or non-standard environment questions, escalate to the solution architect team
4. Installation support is included with all paid plans; do not dismiss installation questions

## 12. Account information updates

Customers can update account information through the account settings portal:

- **Company name and billing address**: Update in the Billing section
- **Contact email and phone**: Update in Account Settings
- **Password**: Direct to the official password-reset flow (never accept password changes via email/chat)
- **Payment method**: Update securely in the Billing section
- **Authorized users and permissions**: Manage in the Team section

For account update requests:

1. First, verify the customer's identity (email address on file, last four digits of account payment method)
2. Direct them to self-service options where possible to reduce support burden
3. For changes that require human verification (account transfer, legal entity changes), escalate to account management
4. Never process account changes based solely on email requests without identity verification

## 13. Bulk discounts and volume pricing

Volume pricing is available for enterprise and high-volume customers:

- **10-49 seats**: Up to 10% discount
- **50-199 seats**: Up to 20% discount
- **200+ seats**: Up to 30% discount; custom pricing available

For discount inquiries:

1. Ask the customer for their expected seat count and contract duration
2. If they qualify for volume pricing (10+ seats), generate a quote or connect them with the sales team
3. For existing customers requesting mid-contract discounts, escalate to account management with their current contract terms
4. Do not negotiate discounts unilaterally; pricing authority rests with the sales or account management teams
5. Enterprise contracts may have custom terms; review existing documentation before discussing discounts

## 14. Delivery and onboarding timelines

Acme Cloud is a SaaS platform with immediate deployment:

- **Cloud deployments**: Activation within 5–15 minutes after subscription confirmation
- **Hybrid deployments**: Initial setup in 1–3 business days after infrastructure requirements are verified
- **On-premises deployments**: 2–4 weeks depending on network configuration and security requirements
- **Onboarding support**: New customers receive 30 days of included onboarding assistance

When customers ask about timelines:

1. Specify the deployment type (cloud, hybrid, or on-premises) because timelines vary significantly
2. Explain prerequisites that might affect timeline (network setup, security approval, hardware procurement)
3. Do not promise specific activation times; say "typically" or "usually" and provide the range above
4. For delays, investigate the root cause and provide a clear revised timeline with explanation
5. Escalate infrastructure issues or repeated delays to the operations team

## 15. Data protection rules

Never ask customers to provide:

- Full payment-card numbers
- CVV or security codes
- Passwords
- One-time passwords
- Recovery codes
- Private API keys

Use transaction IDs, ticket IDs, account email addresses, or the last four digits of a payment method only when necessary and permitted.

If a customer sends sensitive information, do not repeat it in the response. Ask them to remove it from future messages and escalate according to the security process.

## 16. Escalation requirements

Escalate to a human specialist when:

- Two completed charges may exist and the refund workflow is unclear.
- The customer disputes an account ownership decision.
- There is suspected account takeover or payment fraud.
- A payment provider reports conflicting statuses.
- The handbook does not contain enough information to answer safely.
- The customer requests an exception to a policy.

An escalation should include the ticket ID, a short factual summary, relevant transaction IDs, timestamps with timezone, and the troubleshooting steps already completed. Do not include passwords, full card numbers, CVV values, one-time passwords, or private keys.

## 17. Response principles

Support responses should be:

- Polite and concise
- Based only on verified information
- Clear about the next action
- Honest about uncertainty
- Free of invented timelines, refunds, or policy exceptions

If the handbook does not answer the customer’s question, say that a human agent must investigate rather than guessing.
