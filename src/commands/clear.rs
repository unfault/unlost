pub(crate) fn run(path: String, yes: bool) -> anyhow::Result<()> {
    crate::workspace::clear_workspace(std::path::Path::new(&path), yes)
}
