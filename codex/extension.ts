---
	boilerplate extension.ts code:
---
	```
// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "oaicodex" is now active!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('oaicodex.helloWorld', () => {
		// The code you place here will be executed every time your command is executed
		// Display a message box to the user
		vscode.window.showInformationMessage('Hello World from oaicodex!');
	});

	context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
export function deactivate() {}
```

---
	Code to integrate:
---

	```typescriptreact

import * as vscode from 'vscode';
import * as codebase from "./codebase";
import { LuaDebugAdapterDescriptorFactory } from "./debugger/DebugAdapterDescriptorFactory";
import { LuaDebugConfigurationProvider } from "./debugger/DebugConfigurationProvider";
import { LuaDebugSession } from "./debugger/DebugSession";
import * as mixpanel from "./mixpanel";
import * as pyrobot from "./pyrobot";
import * as webview from "./webview";

export function activate( context: vscode.ExtensionContext ) {
	console.log( process.versions );

	context.subscriptions.push(
		vscode.commands.registerCommand( "getContext", () => context )
	);

	const webView = new webview.WebView( context.extensionUri );
	const config = vscode.workspace.getConfiguration( "easycode" );
	webView.setAuthenticationInfo( { apiKey: config.get( "apiKey" )! } );
	webView.setSettings( {
		selectedInsideCodeblock: config.get( "selectedInsideCodeblock;" ) || false,
		copyOnClick: config.get( "copyOnClick" ) || false,
		maxTokens: config.get( "maxTokens" ) || 500,
		temperature: config.get( "temperature" ) || 0,
		model: config.get( "model" ) || "ChatGPT",
		userEmail: config.get( "userEmail" ) || "",
	} );

	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider(
			webview.WebView.viewType,
			webView,
			{ webviewOptions: { retainContextWhenHidden: true } }
		)
	);

	const ask = ( name: string ) => {
		const prompt = vscode.workspace.getConfiguration( "easycode" ).get( name )!;
		webView.ask( prompt, true );
	};

	context.subscriptions.push(
		vscode.commands.registerCommand( "easycode.ask", () => {
			vscode.window
				.showInputBox( { prompt: "What is your question?" } )
				.then( ( prompt ) => {
					webView
						.getMixPanel()
						?.track( "Ask EasyCode", {
							distinct_id: webView.getMachineId(),
							prompt: prompt,
						} );
					webView.ask( prompt!, true );
				} );
		} ),
		vscode.commands.registerCommand( "easycode.writeCode", () => {
			vscode.window
				.showInputBox( { prompt: "What code would you like to write" } )
				.then( ( prompt ) => {
					webView
						.getMixPanel()
						?.track( "Write Code", {
							distinct_id: webView.getMachineId(),
							prompt: prompt,
						} );
					webView.ask( prompt!, true );
				} );
		} ),
		vscode.commands.registerCommand( "easycode.indexCodebase", () => {
			webView
				.getMixPanel()
				?.track( "Index Codebase", {
					distinct_id: webView.getMachineId(),
				} );
			webView.indexCodebase( false );
		} ),
		vscode.commands.registerCommand( "easycode.explainFileFlow", () => {
			webView
				.getMixPanel()
				?.track( "Explain File Flow", {
					distinct_id: webView.getMachineId(),
				} );
			ask( "promptPrefix.explainFileFlow" );
		} ),
		vscode.commands.registerCommand( "easycode.explainCodeFlow", () => {
			webView
				.getMixPanel()
				?.track( "Explain Code Flow", {
					distinct_id: webView.getMachineId(),
				} );
			ask( "promptPrefix.explainCodeFlow" );
		} ),
		vscode.commands.registerCommand( "easycode.explainMethodFlow", () => {
			webView
				.getMixPanel()
				?.track( "Explain Method Flow", {
					distinct_id: webView.getMachineId(),
				} );
			ask( "promptPrefix.explainMethodFlow" );
		} ),
		vscode.commands.registerCommand( "easycode.explain", () => {
			webView
				.getMixPanel()
				?.track( "Explain Selection", {
					distinct_id: webView.getMachineId(),
				} );
			ask( "promptPrefix.explain" );
		} ),
		vscode.commands.registerCommand( "easycode.explainStackTrace", () => {
			webView
				.getMixPanel()
				?.track( "Explain Stack Trace", {
					distinct_id: webView.getMachineId(),
				} );
			webView.explainStackTrace();
		} ),
		vscode.commands.registerCommand( "easycode.refactor", () => ask( "promptPrefix.refactor" ) ),
		vscode.commands.registerCommand( "easycode.optimize", () => ask( "promptPrefix.optimize" ) ),
		vscode.commands.registerCommand( "easycode.findProblems", () =>
			ask( "promptPrefix.findProblems" )
		),
		vscode.commands.registerCommand( "easycode.documentation", () =>
			ask( "promptPrefix.documentation" )
		),
	);

	vscode.workspace.onDidChangeConfiguration( ( e ) => {
		if ( e.affectsConfiguration( "easycode.apiKey" ) ) {
			const config = vscode.workspace.getConfiguration( "easycode" );
			webView.setAuthenticationInfo( { apiKey: config.get( "apiKey" )! } );
			console.log( "API key changed" );
		} else if ( e.affectsConfiguration( "easycode.selectedInsideCodeblock" ) ) {
			const config = vscode.workspace.getConfiguration( "easycode" );
			webView.setSettings( {
				selectedInsideCodeblock:
					config.get( "selectedInsideCodeblock" ) || false,
			} );
		} else if ( e.affectsConfiguration( "easycode.copyOnClick" ) ) {
			const config = vscode.workspace.getConfiguration( "easycode" );
			webView.setSettings( { copyOnClick: config.get( "copyOnClick" ) || false } );
		} else if ( e.affectsConfiguration( "easycode.maxTokens" ) ) {
			const config = vscode.workspace.getConfiguration( "easycode" );
			webView.setSettings( { maxTokens: config.get( "maxTokens" ) || 500 } );
		} else if ( e.affectsConfiguration( "easycode.temperature" ) ) {
			const config = vscode.workspace.getConfiguration( "easycode" );
			webView.setSettings( { temperature: config.get( "temperature" ) || 0 } );
		} else if ( e.affectsConfiguration( "easycode.model" ) ) {
			const config = vscode.workspace.getConfiguration( "easycode" );
			webView.setSettings( { model: config.get( "model" ) || "text-davinci-003" } );
		}
	} );

	vscode.debug.registerDebugAdapterTrackerFactory( "*", {
		createDebugAdapterTracker: ( session: vscode.DebugSession ) => ( {
			onDidSendMessage: ( e: vscode.DebugSessionCustomEvent ) => {
				if ( e.event === "output" && e.body && e.body.category === "stderr" ) {
					if ( session instanceof LuaDebugSession ) {
						webView.setErrorInfo( e.body );
						console.log( e );
					}
				}
			},
		} ),
	} );

	const provider = new LuaDebugConfigurationProvider();
	context.subscriptions.push(
		vscode.debug.registerDebugConfigurationProvider( "lua", provider )
	);

	const factory = new LuaDebugAdapterDescriptorFactory();
	context.subscriptions.push(
		vscode.debug.registerDebugAdapterDescriptorFactory( "lua", factory )
	);

	context.subscriptions.push( factory );

	context.subscriptions.push( pyrobot.PyRobot.getInstance( context ) );

	context.subscriptions.push( codebase.CodebaseIndexer.getInstance( context ) );

	context.subscriptions.push( mixpanel.MixPanel.getInstance( context ) );
}

/*
# Optional: Define custom prompt to start the conversation
# For example: "How can I help you today?"
#prompt = "Hello! How can I help you today?"

# OPTIONAL: Add any custom response options to the system here
#response_options = ['Option A', 'Option B', 'Option C']

# Start the convo
#assistant.start_conversation(prompt=prompt, response_options=response_options)

assistant.start_conversation()<|im_sep|>
 */
```
---
	New extension.js: