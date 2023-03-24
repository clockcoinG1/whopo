/**
* Returns the list of files in the workspace
* @param extension
*/
async function getFilesInWorkspace(extension: string): Promise<vscode.Uri[]> {
    const settings = vscode.workspace.getConfiguration("search");
    const excludeFolders = settings.get<string[]>("excludeFolders") || ["**/node_modules/**", "**/bower_components/**"];
    const includeFiles = settings.get<string[]>("includeFiles") || [`**/*${extension}`];

    const folders = vscode.workspace.workspaceFolders;
    if (folders && folders.length > 0) {
        const promises = folders.map(folder => {
            const options: vscode.GlobPattern = {
                base: folder.uri.fsPath,
                pattern: excludeFolders.concat(includeFiles).join(",")
            };
            return vscode.workspace.findFiles(options);
        });
        const uris = await Promise.all(promises);
        return uris.flat();
    }
    return [];
}

/**
* Returns all symbols (of type 'kind') from the given Uri
* @param kind - the kind of symbol to search for
* @param uri
*/
async function findSymbols(kind: vscode.SymbolKind[], uri: vscode.Uri): Promise<vscode.DocumentSymbol[] | undefined> {
    const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>("vscode.executeDocumentSymbolProvider", uri);
    if (!symbols) return;
    return symbols.filter(symbol => kind.includes(symbol.kind));
}

/**
* Given a list of symbols, returns only symbols that are not ignored by the settings
* @param symbols - a list of symbols
*/
function filterIgnoredSymbols(symbols: vscode.DocumentSymbol[]): vscode.DocumentSymbol[] {
    const ignoredSettings: string[] = vscode.workspace.getConfiguration("separators", r.window.activeTextEditor?.document).get("functions.ignoreCallbackInline") || []; // ['on', 'off', 'yes', 'no', 'true', 'false', 'null', 'undefined', 'NaN'];
    return symbols.filter(symbol => {
        const fullSymbolName = symbol.name;
        const splitName = fullSymbolName.split(".");
        splitName.push(fullSymbolName);
        splitName.push(fullSymbolName.replace(/\./g, ""));
        return !ignoredSettings.includes(fullSymbolName) && !ignoredSettings.includes(splitName[0]) && !ignoredSettings.includes(splitName[splitName.length - 1]);
    });
}

/**
* Returns a list of symbols along with their name, range, and text
* @param kind - the type of symbol to search for
* @param uri
*/
async function getSymbolProperties(kind: vscode.SymbolKind[], uri: vscode.Uri): Promise<Map<string, { name: string, range: vscode.Range }>> {
    const symbols = await findSymbols(kind, uri);
    let properties: Map<string, { name: string, range: vscode.Range }> = new Map();
    if (symbols) {
        symbols.forEach(symbol => {
            filterIgnoredSymbols([symbol]).forEach(filteredSymbol => {
                const fullSymbolName = symbol.name;
                const splitName = fullSymbolName.split(".");
                const methodOrPropertyName = splitName[splitName.length - 1];
                properties.set(methodOrPropertyName + " - " + uri.path, {
                    name: methodOrPropertyName,
                    range: filteredSymbol.range
                });
            });

        });
    }
    return properties;
}

/**
* Returns a list of methods from the codebase
*/
async function getMethodsInCodebase() {
    const folder = vscode.workspace.workspaceFolders;
    if (!folder) return undefined;
    const files = await getFilesInWorkspace(".ts,.js,.jsx,.tsx,.cs,.lua,.py,.java,.html");
    const symbols: vscode.DocumentSymbol[] = [];
    for (const file of files) {
        const document = await vscode.workspace.openTextDocument(file);
        const uri = document.uri;
        const documentSymbols = await findSymbols([vscode.SymbolKind.Function, vscode.SymbolKind.Method, vscode.SymbolKind.Constructor], uri);
        if (documentSymbols) {
            symbols.push(...documentSymbols);
        }
    };
    const symbolProperties = new Map();
    for (const symbol of symbols) {
        const filteredSymbols = filterIgnoredSymbols([symbol]);
        filteredSymbols.forEach(filteredSymbol => {
            const splitName = filteredSymbol.name.split(".");
            const methodOrPropertyName = splitName[splitName.length - 1];
            if (filteredSymbol.range) {
                symbolProperties.set(methodOrPropertyName + " - " + symbol.uri.path, {
                    name: methodOrPropertyName,
                    range: filteredSymbol.range
                });
            }
        });
    }

    return Array.from(symbolProperties.keys()).map(key => {
        return {
            methodText: symbolProperties.get(key)?.range ? (await vscode.workspace.openTextDocument(symbolProperties.get(key)?.range.start.line))?.getText() : "No text found.",
            methodName: key.split(" - ")[0],
            fileUri: key.split(" - ")[1]
        };
    });
}

/**
* Returns a mapping of reference methods to their location in the code
* @param methods
*/
async function getReferencesToMethods(methods: { methodName: string, methodText: string, fileUri: string }[]) {
    const result: { methodName: string, ReferenceMethodFileMapping: Map<vscode.Uri, { name: string, range: vscode.Range }> }[] = [];
    for (const method of methods) {
        const tokens = lexer[method.fileUri + " - " + method.methodName];
        if (!tokens) continue;
        const positions = await vscode.commands.executeCommand<vscode.Range[]>("vscode.executeReferenceProvider", vscode.Uri.parse(method.fileUri), tokens[0].range.start);
        if (!positions) continue;
        let referenceMethodFileMapping: Map<vscode.Uri, { name: string, range: vscode.Range }> = new Map();
        for (const position of positions) {
            if (position.uri.path === method.fileUri.split(":")[0]) continue;
            referenceMethodFileMapping.set(position.uri, {
                name: method.methodName,
                range: position.range
            });
        }
        result.push({
            methodName: method.methodName,
            ReferenceMethodFileMapping: referenceMethodFileMapping
        });
    }
    return result;
}

/**
* Given a list of code references, sends each one to the OpenAI GPT-3 API and appends the result to the output window
* @param references
*/
async function lookupReferencesWithGPT(references: { methodName: string, ReferenceMethodFileMapping: Map<vscode.Uri, { name: string, range: vscode.Range }> }[]) {
    let gptTexts: string[] = [];
    let referenceMappings: { methodName: string, ReferenceMethodFileMapping: Map<vscode.Uri, { name: string, range: vscode.Range }> }[] = [];
    let textToUriMapping: { [text: string]: vscode.Uri } = {};
    for (const reference of references) {
        for (const [uri, range] of reference.ReferenceMethodFileMapping) {
            const fileText = (await vscode.workspace.openTextDocument(uri)).getText(range);
            const prompt = `Explain how the ${reference.methodName} method is used by ${range.name}: `;
            const text = `${fileText}\n\n${reference.methodText}`;
            // only send to openai if the text length is within the API limit
            if (l.encode(text).length < 8193 - (this._settings.maxTokens || 1024) - 1000) {
                gptTexts.push(text);
                textToUriMapping[text] = uri;
                referenceMappings.push(reference);
            }
        }
    }
    if (!gptTexts.length) {
        this._view?.webview.postMessage({ type: "addResponse", value: "No references found." });
        return;
    }
    const stream = this._settings.model === "ChatGPT";
    try {
        let response;
        if (!this._openai) return;
        const model = this._settings.model || "code-davinci-002";
        const multipleRequests = gptTexts.length > 1;
        const maxTokens = this._settings.maxTokens || 1024;
        const remainingTokens = this._openai.remainingTokens(model);
        const numRequests = Math.ceil(gptTexts.length / (remainingTokens / maxTokens));
        for (let i = 0; i < numRequests; i++) {
            this._view?.webview.postMessage({ type: "addResponse", value: `Submitting request ${i + 1}/${numRequests} to OpenAI. Remaining tokens: ${remainingTokens}.` });
            let currentTokenCount = 0;
            let textsToSend: string[] = [];
            let mappings: { methodName: string, ReferenceMethodFileMapping: Map<vscode.Uri, { name: string, range: vscode.Range }> }[] = [];
            let currentText: string;
            while (currentTokenCount + l.encode(gptTexts[0]).length < remainingTokens && gptTexts.length > 0) {
                currentText = gptTexts.shift() ?? "";
                currentTokenCount += l.encode(currentText).length;
                textsToSend.push(currentText);
                mappings.push(referenceMappings.shift() || { ReferenceMethodFileMapping: new Map(), methodName: "" });
            }
            const prompt = this._settings.model === "ChatGPT" ? textsToSend[0] : this._fullPrompt;
            response = await this._openai.createCompletion(
                {
                    model: model,
                    prompt: prompt,
                    temperature: this._settings.temperature,
                    max_tokens: maxTokens,
                    stream: stream,
                    stop: ["\nUSER: ", "\nUSER", "\nASSISTANT"]
                }
            );
            const choices = response?.data?.choices || [];
            for (let i = 0; i < choices.length; i++) {
                const message = choices[i].text;
                const requestIndex = multipleRequests ? i : 0;
                const referenceMapping = mappings[requestIndex].ReferenceMethodFileMapping;
                if (!referenceMapping) continue;
                const uri = Array.from(referenceMapping.keys())[0];
                const methodName = referenceMapping.values().next().value.name;
                if (!methodName || !uri) continue;
                let currentText = textsToSend[requestIndex];
                while (currentText.startsWith(" ") || currentText.startsWith("\n")) {
                    currentText = currentText.slice(1);
                }
                l.getPromptsFromString(currentText).forEach(prompt => {
                    message.replace(prompt, "\n" + prompt);
                });
                const outputValue = `${Array.from(textToUriMapping).find((_text, uri) => uri === uri)?.[0]} - ${methodName}\n\n${message}\n\n---\n`;
                this._view?.webview.postMessage({ type: "appendResponseWithReference", value: outputValue, uniqueMethodKey: `${Array.from(textToUriMapping).find((_text, uri) => uri === uri)?.[0]} - ${methodName}` });
                const tokenCount = l.encode(message).length;
                this.getMixPanel()?.people.increment(this.getMachineId(), "OpenAI Output Tokens Used", tokenCount);
                this.getMixPanel()?.people.increment(this.getMachineId(), "OpenAI Cost ($0.02/1000 tokens)", tokenCount / 1000 * 0.02);
            }
        }
    } catch (err) {
        let error = "";
        if (err.response) { console.log(err.response.status); console.log(err.response.data); error = `${err.response.status};${err.response.data.message} ` }
        else { console.log(err.message); error = err.message }
        this._view?.webview.postMessage({ type: "addResponse", value: `\n\n---\n[ERROR] ${error}` });
    }
}

/**
* Indexes the codebase
* @param e
*/
async function indexCodebase(e: vscode.ExtensionContext | undefined) {
    let folders = vscode.workspace.workspaceFolders;
    if (folders) {
        const options: vscode.OpenDialogOptions = {
            defaultUri: folders[0].uri,
            canSelectMany: false,
            openLabel: "Select",
            canSelectFiles: false,
            canSelectFolders: true
        };
        let selectedProjectsFolder = "";
        if ( await vscode.window.showInformationMessage( "This feature requires you to first index your codebase. Please select your project folder.", { modal: true } ) ) {
            const uris = await vscode.window.showOpenDialog(options);
            if (uris && uris.length > 0) {
                selectedProjectsFolder = uris[0].fsPath;
            }
            this._view?.webview.postMessage({ type: "clearResponse" });
            this._view?.webview.postMessage({ type: "clearReferences" });
            const workspacePath = folders[0].uri.fsPath;
            selectedProjectsFolder = selectedProjectsFolder.split(workspacePath)[1].substring(1);
            if (selectedProjectsFolder != "") {
                selectedProjectsFolder += "/";
            }
            const excludeFolders = ["**/node_modules/**", "**/dist/**"];
            const includeFiles = [`**/*.{java,py,ts,js,tsx,jsx,html,cs,lua}`];
            const files = await getFilesInWorkspace(includeFiles[0]);
            let foundSymbols = new Map();
            for (let i = 0; i < Math.min(files.length, 500); i++) {
                this._view?.webview.postMessage({ type: "addResponse", value: `Indexing Codebase 1/2. Remaining Tasks: ${Math.min(files.length, 500) - i}` });
                const file = files[i];
                if (file.path.includes("min.js") || file.path.includes("min.ts") || file.path.includes("test.ts") || file.path.includes("spec.ts")) {
                    continue;
                }
                if (x(file.path.split(".")) == "cs") {
                    await vscode.window.showTextDocument(file);
                }
                const symbols = await findSymbols([vscode.SymbolKind.Function, vscode.SymbolKind.Method, vscode.SymbolKind.Constructor], file);
                if (symbols) {
                    for (let i = 0; i < symbols.length; i++) {
                        const symbol = symbols[i];
                        if (filterIgnoredSymbols([symbol]).length === 0) {
                            continue;
                        }
                        const methodOrPropertyName: string = symbol.name.split(".").pop() || symbol.name;
                        const text = (await vscode.workspace.openTextDocument(file)).getText(symbol.range);
                        const tokens = l.getPromptsFromString(text);
                        if (l.encode(text).length < (8193 - ((this._settings.maxTokens || 1024) - 1000) / 2)) {
                            for (let i = 0; i < tokens.length; i++) {
                                const token = tokens[i];
                                foundSymbols.set(token, methodOrPropertyName + " - " + file.path + " - " + symbol.range.start.line);
                            }
                        }
                    }
                }
            }
            const textToEmbed = Array.from(foundSymbols.keys());
            let error = "";
            if (this._openai) {
                try {
                    let numRequests = 1;
                    let embeddings = new Map();
                    let textsToSend = textToEmbed;
                    while (textsToSend.length > 0) {
                        this._view?.webview.postMessage({ type: "addResponse", value: `Indexing Codebase 2/2. Remaining Tasks: ${textsToSend.length}` });
                        const currentTokenCount = l.encode(textsToSend[0]).length;
                        const remainingTokens = this._openai.remainingTokens("text-embedding-ada-002");
                        const numRequestsForCurrentBatch = Math.ceil(currentTokenCount / remainingTokens);
                        if (numRequestsForCurrentBatch > numRequests) {
                            numRequests = numRequestsForCurrentBatch;
                        }
                        let currentRequestTokens = 0;
                        let currentTexts: string[] = [];
                        while (currentRequestTokens < remainingTokens && textsToSend.length > 0) {
                            const textToSend = textsToSend.splice(0, 1)[0];
                            const tokenCount = l.encode(textToSend).length;
                            if (currentRequestTokens + tokenCount > remainingTokens) {
                                continue;
                            }
                            currentRequestTokens += tokenCount;
                            currentTexts.push(textToSend);
                        }
                        const input = { model: "text-embedding-ada-002", input: currentTexts };
                        const response = await this._openai.createEmbedding(input);
                        if (response?.data?.data?.length) {
                            for (let i = 0; i < response.data.data.length; i++) {
                                const embedding = response.data.data[i].embedding;
                                embeddings.set(currentTexts[i], embedding);
                            }
                        }
                    }
                    const embeddingsStr = JSON.stringify(Array.from(embeddings));
                    e?.workspaceState.update("embeddingMap", embeddingsStr);
                } catch (err) {
                    if (err.response) { console.log(err.response.status); console.log(err.response.data); error = `${err.response.status};${err.response.data.message} ` }
                    else { console.log(err.message); error = err.message }
                }
                if (!error) {
                    this._view?.webview.postMessage({ type: "addResponse", value: "Indexing Complete." });
                    this._view?.webview.postMessage({ type: "hideIndexCodeButton", value: true });
                    this.getMixPanel()?.track("Finished Indexing Codebase", { distinct_id: this.getMachineId() });
                    return;
                }
            } else error = "[ERROR] API token not set, please go to extension settings to set it (read README.md for more info)";
            this._view?.webview.postMessage({ type: "addResponse", value: `\n\n---\n[ERROR] ${error} ` });
        }
    } else {
        this._view?.webview.postMessage({ type: "clearResponse" });
        this._view?.webview.postMessage({ type: "addResponse", value: "You must have a project open before this can be used." });
    }
}