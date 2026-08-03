# Access Control Policy

**Agent-trigger:** Who may hold which credentials and for how long: least-privilege grants, service accounts, key rotation, and revoking access when someone leaves. Load before creating a role, granting a permission, issuing a token or service account, and when access must be revoked or rotated.

Access is granted to a role, never to a person directly, and every role states the smallest permission
set that lets its holder finish their work. A permission granted "temporarily" carries an expiry at the
moment it is granted or it is not temporary.

Long-lived static credentials are prohibited where a federated short-lived token is available. Where one
is not, the key rotates every ninety days on a schedule that does not depend on anyone remembering.

Departure revokes every grant within one hour, and the revocation is verified rather than assumed.
