// Every route in the generated registry carries `require.resolveWeak(page)`.
// A weak reference still makes the bundler build the page, which defeats lazy
// compilation. Replacing it with the bare path makes the page look "not loaded
// yet" to the client, so it is fetched (and only then compiled) on navigation.
module.exports = function dropWeakRefs(source) {
  return source.replaceAll('require.resolveWeak(', '(');
};
