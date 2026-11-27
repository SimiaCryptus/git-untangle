const VIEWER_TEMPLATE: &str = include_str!("../assets/viewer.html");
const CONFIG_START: &str = "/*CONFIG_START*/";
const CONFIG_END: &str = "/*CONFIG_END*/";

/// Renders a self-contained HTML viewer with embedded `graph_json` and safe `</script>` escaping.
pub fn render_viewer_html(src_filename: &str, short_id: &str, subject: &str, graph_json: &str) -> String {
    let raw_title = format!("git-untangle: {short_id} — {subject}");
    let title: String = raw_title.chars().take(200).collect();

    let src_escaped = serde_json::to_string(src_filename).unwrap_or_else(|_| "\"graph.json\"".to_owned());
    let title_escaped = serde_json::to_string(&title).unwrap_or_else(|_| "\"git-untangle\"".to_owned());

    let config_payload = format!(
        "{{\n  \"src\": {src_escaped},\n  \"title\": {title_escaped},\n  \"data\": {graph_json}\n}}"
    )
    .replace('<', "\\u003c");

    if let (Some(start), Some(end)) = (
        VIEWER_TEMPLATE.find(CONFIG_START),
        VIEWER_TEMPLATE.find(CONFIG_END),
    ) {
        if end > start {
            let prefix_end = start + CONFIG_START.len();
            let mut html = String::with_capacity(VIEWER_TEMPLATE.len() + config_payload.len());
            html.push_str(&VIEWER_TEMPLATE[..prefix_end]);
            html.push_str(&config_payload);
            html.push_str(&VIEWER_TEMPLATE[end..]);
            return html;
        }
    }
    VIEWER_TEMPLATE.to_owned()
}
