# example
from zlutils.framework.ep.app import App


class MainApp(App):

    def on_running(self):
        self.plugin_manager.publishEvent('pmprint')


if __name__ == "__main__":

    plugin_dir = 'Plugins'  # default
    mainapp = MainApp(plugin_dir=plugin_dir)
    print('lifecycle_states',mainapp.lifecycle_states)
    mainapp.run()
