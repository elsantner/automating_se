static int samldb_fill_foreignSecurityPrincipal_object(struct samldb_ctx *ac)
{
	struct ldb_context *ldb;
	const struct ldb_val *rdn_value;
	struct dom_sid *sid;
	int ret;

	ldb = ldb_module_get_ctx(ac->module);

	sid = samdb_result_dom_sid(ac->msg, ac->msg, "objectSid");
	if (sid == NULL) {
		rdn_value = ldb_dn_get_rdn_val(ac->msg->dn);
		if (rdn_value == NULL) {
			return ldb_operr(ldb);
		}
		sid = dom_sid_parse_talloc(ac->msg,
					   (const char *)rdn_value->data);
		if (sid == NULL) {
			ldb_set_errstring(ldb,
					  "samldb: No valid SID found in ForeignSecurityPrincipal CN!");
			return LDB_ERR_CONSTRAINT_VIOLATION;
		}
		if (! samldb_msg_add_sid(ac->msg, "objectSid", sid)) {
			return ldb_operr(ldb);
		}
	}

	/* finally proceed with adding the entry */
	ret = samldb_add_step(ac, samldb_add_entry);
	if (ret != LDB_SUCCESS) return ret;

	return samldb_first_step(ac);
}
